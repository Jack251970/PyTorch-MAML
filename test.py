import random

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import models
import utils
from utils.arguments import parse_launch_parameters
from utils.basic import acquire_device, get_data


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    ##### Dataset #####

    # meta-test
    dataset, loader = get_data(args, 'test')
    utils.log('meta-test set: {} (x{})'.format(dataset[0][0].shape, len(dataset)))

    ##### Model #####

    if args.use_gpu:
        ckpt = torch.load(args.load)
    else:
        ckpt = torch.load(args.load, map_location=torch.device('cpu'))
    model = models.load(ckpt, args).to(device)

    if args.efficient:
        model.go_efficient()

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    ##### Evaluation #####

    model.eval()
    aves_va = utils.AverageMeter()
    va_lst = []

    for epoch in range(1, args.train_epochs + 1):
        for data in tqdm(loader, leave=False):
            x_shot, x_query, y_shot, y_query = data
            x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
            x_query, y_query = x_query.cuda(), y_query.cuda()

            logits = model(x_shot, x_query, y_shot, meta_train=False)
            logits = logits.view(-1, args.n_way)
            labels = y_query.view(-1)

            pred = torch.argmax(logits, dim=1)
            acc = utils.compute_acc(pred, labels)
            aves_va.update(acc, 1)
            va_lst.append(acc)

        print('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
            epoch, aves_va.item() * 100,
                   utils.mean_confidence_interval(va_lst) * 100))


if __name__ == '__main__':
    args = parse_launch_parameters()
    device = acquire_device(args)
    main()
