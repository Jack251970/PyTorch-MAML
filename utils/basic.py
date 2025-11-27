import math
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.penmanshiel_data import DatasetPenmanshiel
from datasets.wind_data import DatasetWind


def acquire_device(args):
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            args.gpu) if not args.use_multi_gpu else args.devices
        device = torch.device('cuda:{}'.format(args.gpu))
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        print('Use GPU: cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    else:
        raise NotImplementedError

    if epoch in lr_adjust.keys():
        # get lr in dictionary
        lr = lr_adjust[epoch]

        # update lr in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return 'Updating learning rate to {}'.format(lr)
    else:
        return None


def get_wind_data(args, data_flag):
    # get data information
    timeenc = 0 if args.embed != 'timeF' else 1
    pin_memory = args.pin_memory

    shuffle_flag = False if data_flag == 'test' else True
    drop_last = True
    batch_size = args.n_episode  # In meta learning, each batch contains multiple tasks

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
        pin_memory=pin_memory
    )
    return data_set, data_loader


def get_penmanshiel_data(args, data_flag):
    # get data information
    timeenc = 0 if args.embed != 'timeF' else 1
    pin_memory = args.pin_memory

    shuffle_flag = False if data_flag == 'test' else True
    drop_last = True
    batch_size = args.n_episode  # In meta learning, each batch contains multiple tasks

    data_set = DatasetPenmanshiel(
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
        pin_memory=pin_memory
    )
    return data_set, data_loader


def handle_features(args):
    data = pd.read_csv(os.path.join(args.root_path, args.data_path))
    feature_length = len(data.columns) - 1
    args.enc_in = feature_length
    args.dec_in = feature_length
    args.c_out = feature_length
