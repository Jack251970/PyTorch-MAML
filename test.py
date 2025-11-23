import random

import torch
import numpy as np
from torch import nn
from tqdm import tqdm

import models
import utils
from utils.arguments import parse_meta_launch_parameters
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

    args.load = 'save/meta_wind_5_way_5_shot_2025_11_23_16_01_05/min-vl.pth'
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

    loss_fn = nn.MSELoss()

    for data in tqdm(loader, leave=False):
        x_shot, x_query, y_shot, y_query = data
        x_shot, y_shot = x_shot.to(device).float(), y_shot.to(device).float()
        x_query, y_query = x_query.to(device).float(), y_query.to(device).float()

        logits = model(x_shot, x_query, y_shot, meta_train=False)

        f_dim = -1 if args.features == 'MS' else 0
        preds = logits[..., f_dim]
        labels = y_query[..., f_dim]

        loss = loss_fn(preds, labels)
        aves_va.update(loss.item(), 1)
        va_lst.append(loss.item())

    print('test: loss={:.2f} +- {:.2f} (%)'.format(aves_va.item() * 100,
                                                   utils.mean_confidence_interval(va_lst) * 100))


if __name__ == '__main__':
    args = parse_meta_launch_parameters()
    device = acquire_device(args)
    main()
