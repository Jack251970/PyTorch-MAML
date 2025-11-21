import argparse
import os
import random
import time
from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.optimizers as optimizers
from utils.arguments import parse_launch_parameters

from datasets.wind_data import DatasetWind


def get_data(data_flag):
    # get data information
    timeenc = 0 if args.embed != 'timeF' else 1
    pin_memory = args.pin_memory

    shuffle_flag = False if data_flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size

    data_set = DatasetWind(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=data_flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        scaler=args.scaler,
        timeenc=timeenc,
        freq=args.freq,
        lag=args.lag,
        seasonal_patterns=args.seasonal_patterns
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    return data_set, data_loader


def main(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    ckpt_name = args.name
    if ckpt_name is None:
        ckpt_name = config['encoder']
        ckpt_name += '_' + config['dataset'].replace('meta-', '')
        ckpt_name += '_{}_way_{}_shot'.format(
            config['train']['n_way'], config['train']['n_shot'])
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
    yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

    ##### Dataset #####

    # meta-train
    # train_set = datasets.make(config['dataset'], **config['train'])
    # utils.log('meta-train set: {} (x{}), {}'.format(
    #     train_set[0][0].shape, len(train_set), train_set.n_classes))
    # train_loader = DataLoader(
    #     train_set, config['train']['n_episode'],
    #     collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)
    train_set, train_loader = get_data('train')
    utils.log('meta-train set: {} (x{}), {}'.format(
        train_set[0][0].shape, len(train_set), train_set.n_classes))

    # meta-val
    # eval_val = False
    # if config.get('val'):
    #     eval_val = True
    #     val_set = datasets.make(config['dataset'], **config['val'])
    #     utils.log('meta-val set: {} (x{}), {}'.format(
    #         val_set[0][0].shape, len(val_set), val_set.n_classes))
    #     val_loader = DataLoader(
    #         val_set, config['val']['n_episode'],
    #         collate_fn=datasets.collate_fn, num_workers=1, pin_memory=True)
    eval_val = False
    if config.get('val'):
        eval_val = True
        val_set, val_loader = get_data('val')
        utils.log('meta-val set: {} (x{}), {}'.format(
            val_set[0][0].shape, len(val_set), val_set.n_classes))

    ##### Model and Optimizer #####

    inner_args = utils.config_inner_args(config.get('inner_args'))
    if config.get('load'):
        ckpt = torch.load(config['load'], map_location=torch.device('cpu'))  # load parameters from a checkpoint
        model = models.load(ckpt, args)
        optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
        start_epoch = ckpt['training']['epoch'] + 1
        max_va = ckpt['training']['max_va']
    else:
        model = models.make(args)
        optimizer, lr_scheduler = optimizers.make(
            config['optimizer'], model.parameters(), **config['optimizer_args'])
        start_epoch = 1
        max_va = 0.

    if args.efficient:
        model.go_efficient()

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

    ##### Training and evaluation #####

    # 'tl': meta-train loss
    # 'ta': meta-train accuracy
    # 'vl': meta-val loss
    # 'va': meta-val accuracy
    aves_keys = ['tl', 'ta', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    # 使用多个epoch不断训练寻找最佳的初始参数，使得模型在少量梯度更新后能有较好的表现
    # 一个train_loader中所提供的多个Batch of tasks不一定能确保最终的收敛，因此在工程中应当用多个epoch来训练模型
    for epoch in range(start_epoch, config['epoch'] + 1):
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
            x_shot, x_query, y_shot, y_query = data
            x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
            x_query, y_query = x_query.cuda(), y_query.cuda()

            if inner_args['reset_classifier']:
                if config.get('_parallel'):
                    model.module.reset_classifier()
                else:
                    model.reset_classifier()

            logits = model(x_shot, x_query, y_shot, inner_args, meta_train=True)
            logits = logits.flatten(0, 1)
            labels = y_query.flatten()

            pred = torch.argmax(logits, dim=-1)
            acc = utils.compute_acc(pred, labels)
            loss = F.cross_entropy(logits, labels)
            aves['tl'].update(loss.item(), 1)
            aves['ta'].update(acc, 1)

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
                x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
                x_query, y_query = x_query.cuda(), y_query.cuda()

                if inner_args['reset_classifier']:
                    if config.get('_parallel'):
                        model.module.reset_classifier()
                    else:
                        model.reset_classifier()

                logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
                logits = logits.flatten(0, 1)
                labels = y_query.flatten()

                pred = torch.argmax(logits, dim=-1)
                acc = utils.compute_acc(pred, labels)
                loss = F.cross_entropy(logits, labels)
                aves['vl'].update(loss.item(), 1)
                aves['va'].update(acc, 1)

        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, avg in aves.items():
            aves[k] = avg.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.end())
        t_elapsed = utils.time_str(timer_elapsed.end())
        t_estimate = utils.time_str(timer_elapsed.end() /
                                    (epoch - start_epoch + 1) * (config['epoch'] - start_epoch + 1))

        # formats output
        log_str = 'epoch {}, meta-train {:.4f}|{:.4f}'.format(
            str(epoch), aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', meta-val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'meta-val': aves['va']}, epoch)

        log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
        utils.log(log_str)

        # saves model and meta-data
        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'max_va': max(max_va, aves['va']),

            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
            if lr_scheduler is not None else None,
        }
        # ckpt = {
        #     'file': __file__,
        #     'config': config,
        #
        #     'encoder': config['encoder'],
        #     'encoder_args': config['encoder_args'],
        #     'encoder_state_dict': model_.encoder.state_dict(),
        #
        #     'classifier': config['classifier'],
        #     'classifier_args': config['classifier_args'],
        #     'classifier_state_dict': model_.classifier.state_dict(),
        #
        #     'training': training,
        # }

        # 'epoch-last.pth': saved at the latest epoch
        # 'max-va.pth': saved when validation accuracy is at its maximum
        model._save_to_state_dict()
        torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    args = parse_launch_parameters()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    utils.set_gpu(str(args.gpu))
    main(config)
