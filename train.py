import os
import random
import time

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

import models
import utils
from utils.arguments import parse_meta_launch_parameters
from utils.basic import acquire_device, adjust_learning_rate, get_data


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    ckpt_name = 'meta_wind'
    ckpt_name += '_{}_way_{}_shot'.format(args.n_way, args.n_shot)
    if args.tag is None:
        # Use current time as default tag
        t = time.localtime()
        args.tag = time.strftime('%Y_%m_%d_%H_%M_%S', t)
    if args.tag is not None:
        ckpt_name += '_' + args.tag

    ckpt_path = os.path.join('./save', ckpt_name)
    utils.ensure_path(ckpt_path)
    utils.set_log_path(ckpt_path)
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))

    ##### Dataset #####

    # meta-train
    train_set, train_loader = get_data(args, 'train')
    utils.log('meta-train set: {} (x{})'.format(train_set[0][0].shape, len(train_set)))

    # meta-val
    eval_val = True
    val_set, val_loader = get_data(args, 'val')
    utils.log('meta-val set: {} (x{})'.format(val_set[0][0].shape, len(val_set)))

    ##### Model and Optimizer #####

    if args.load is not None:
        if not os.path.exists(args.load):
            raise ValueError('checkpoint {} does not exist.'.format(args.load))
        # load parameters from a checkpoint
        if args.use_gpu:
            ckpt = torch.load(args.load)
        else:
            ckpt = torch.load(args.load, map_location=torch.device('cpu'))
        model = models.load(ckpt, args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['training']['epoch'] + 1
        min_vl = ckpt['training']['min_vl']
    else:
        model = models.make(args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        start_epoch = 1
        min_vl = float('inf')

    if args.efficient:
        model.go_efficient()

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

    ##### Training and evaluation #####

    # 'tl': meta-train loss
    # 'vl': meta-val loss
    aves_keys = ['tl', 'vl']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    loss_fn = nn.MSELoss()

    # 使用多个epoch不断训练寻找最佳的初始参数，使得模型在少量梯度更新后能有较好的表现
    # 一个train_loader中所提供的多个Batch of tasks不一定能确保最终的收敛，因此在工程中应当用多个epoch来训练模型
    for epoch in range(start_epoch, args.train_epochs + 1):
        timer_epoch.start()
        aves = {k: utils.AverageMeter() for k in aves_keys}

        # meta-train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        np.random.seed(epoch)

        # MAML的训练是基于task的，而这里的每个task就相当于普通深度学习模型训练过程中的一条训练数据。
        # 在每一代中，我们会选出多个task（即多个训练数据），然后对每个task进行内环更新，最后再进行一次外环更新。
        # train_loader中每一个data对应一个task，包括支持集和查询集。
        for data in tqdm(train_loader, desc='meta-train', leave=False):  # 获取多个Batch of tasks并进行训练
            x_shot, x_query, y_shot, y_query = data  # [n_episode, n_way * n_shot, H, D]
            x_shot, y_shot = x_shot.to(device).float(), y_shot.to(device).float()
            x_query, y_query = x_query.to(device).float(), y_query.to(device).float()

            # prediction labels
            logits = model(x_shot, x_query, y_shot, meta_train=True)  # [n_episode, n_way * n_shot, H, D]

            f_dim = -1 if args.features == 'MS' else 0
            preds = logits[..., f_dim]
            labels = y_query[..., f_dim]

            loss = loss_fn(preds, labels)
            aves['tl'].update(loss.item(), 1)

            optimizer.zero_grad()
            loss.backward()
            for param in optimizer.param_groups[0]['params']:
                nn.utils.clip_grad_value_(param, 10)
            optimizer.step()

        # meta-val
        # 在每一代训练结束后，我们会在验证集上评估模型的性能，以便观察模型的泛化能力。
        # 也许可以使用Early Stopping来防止过拟合。
        if eval_val:
            model.eval()
            np.random.seed(0)

            for data in tqdm(val_loader, desc='meta-val', leave=False):
                x_shot, x_query, y_shot, y_query = data
                x_shot, y_shot = x_shot.to(device).float(), y_shot.to(device).float()
                x_query, y_query = x_query.to(device).float(), y_query.to(device).float()

                logits = model(x_shot, x_query, y_shot, meta_train=False)

                f_dim = -1 if args.features == 'MS' else 0
                preds = logits[..., f_dim]
                labels = y_query[..., f_dim]

                loss = loss_fn(preds, labels)
                aves['vl'].update(loss.item(), 1)

        adjust_learning_rate(optimizer, epoch, args)

        for k, avg in aves.items():
            aves[k] = avg.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.end())
        t_elapsed = utils.time_str(timer_elapsed.end())
        t_estimate = utils.time_str(timer_elapsed.end() /
                                    (epoch - start_epoch + 1) * (args.train_epochs - start_epoch + 1))

        # formats output
        log_str = 'epoch {}, meta-train {:.4f}'.format(
            str(epoch), aves['tl'])
        writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)

        if eval_val:
            log_str += ', meta-val {:.4f}'.format(aves['vl'])
            writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)

        log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
        utils.log(log_str)

        # saves model and meta-data
        training = {
            'epoch': epoch,
            'min_vl': min(min_vl, aves['vl']),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt = {
            'file': __file__,
            'encoder_state_dict': model.encoder.state_dict(),
            'training': training,
        }

        # 'epoch-last.pth': saved at the latest epoch
        # 'min-vl.pth': saved when validation loss is at its minimum
        torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))

        if aves['vl'] < min_vl:
            min_vl = aves['vl']
            torch.save(ckpt, os.path.join(ckpt_path, 'min-vl.pth'))
            print(f'Saved epoch {epoch} with min-vl {min_vl}')

        writer.flush()


if __name__ == '__main__':
    args = parse_meta_launch_parameters()
    device = acquire_device(args)
    main()
