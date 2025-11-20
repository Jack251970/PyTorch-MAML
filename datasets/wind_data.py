import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer
from torch.utils.data import Dataset

from datasets.uea import interpolate_missing
from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features


def get_search_space():
    default_config = {
        'task_name': {'_type': 'single', '_value': 'long_term_forecast'},
        'is_training': {'_type': 'single', '_value': 1},
        'des': {'_type': 'single', '_value': 'Exp'},
        'use_gpu': {'_type': 'single', '_value': True},
        'embed': {'_type': 'single', '_value': 'timeF'},
        'freq': {'_type': 'single', '_value': 't'},
        'batch_size': {'_type': 'single', '_value': 256},
        'pin_memory': {'_type': 'single', '_value': False},
    }

    dataset_config = {
        'data': {'_type': 'single', '_value': 'custom'},
        'features': {'_type': 'single', '_value': 'MS'},
        'root_path': {'_type': 'single', '_value': './dataset/'},
        'data_path': {'_type': 'single', '_value': 'wind/Zone1/Zone1.csv'},
        'target': {'_type': 'single', '_value': 'wind'},
        'enc_in': {'_type': 'single', '_value': 5},
        'dec_in': {'_type': 'single', '_value': 5},
        'c_out': {'_type': 'single', '_value': 5},
    }

    learning_config = {
        'learning_rate': {'_type': 'single', '_value': 0.0001},
        'train_epochs': {'_type': 'single', '_value': 1},
    }

    period_config = {
        'seq_len': {'_type': 'single', '_value': 96},
        'label_len': {'_type': 'single', '_value': 96},
        'pred_len': {'_type': 'single', '_value': 16},
        'moving_avg': {'_type': 'single', '_value': 25},
    }

    dlinear_config = {
        'd_model': {'_type': 'single', '_value': 128},
        # 'individual': {'_type': 'single', '_value': True},
    }

    return [default_config, dataset_config, learning_config, period_config, dlinear_config]


def read_data(path, target, seq_len, set_type, features, scale, scaler, lag, timeenc, freq, args):
    # read raw data
    df_raw = pd.read_csv(path)

    # check if exist nan
    if df_raw.isnull().values.any():
        df_raw = interpolate_missing(df_raw)

    # df_raw.columns: ['date', ...(other features), target feature]
    cols = list(df_raw.columns)
    if 'date' in cols:
        date_column = 'date'
    else:
        raise NotImplementedError("Make sure your datasets contain the column named 'date'!")

    # remove date and target columns
    cols.remove(target)
    cols.remove(date_column)

    # reorganize df_raw
    df_raw = df_raw[[date_column] + cols + [target]]

    # divide data into train, vali, test parts
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test

    # get boarders of the data
    # set_type: {'train': 0, 'val': 1, 'test': 2}
    border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    border1 = border1s[set_type]
    border2 = border2s[set_type]

    # select features
    if features == 'M' or features == 'MS':
        # select features except the first
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    elif features == 'S':
        # select features only the target
        df_data = df_raw[[target]]
    else:
        raise NotImplementedError

    # apply standard scaler if needed
    if scale:
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)
        data = scaler.transform(df_data.values)
    else:
        data = df_data.values

    # add lag feature
    if lag > 0:
        for i in range(lag, 0, -1):
            label_data = data[:, -1]  # {ndarray: (N,)}
            lag_data = np.concatenate([np.zeros(i), label_data[:-i]], axis=0)  # {ndarray: (N,)}
            data = np.concatenate([lag_data.reshape(-1, 1), data], axis=1)

    # extract date column
    df_stamp = df_raw[[date_column]][border1:border2]
    df_stamp[date_column] = pd.to_datetime(df_stamp.date)

    # encode time
    if timeenc == 0:
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop([date_column], 1).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp[date_column].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    else:
        raise ValueError('timeenc should be 0 or 1!')

    # output data
    data_x = data[border1:border2]
    data_y = data[border1:border2]

    if set_type == 0 and args.augmentation_ratio > 0:
        data_x, data_y, augmentation_tags = run_augmentation_single(data_x, data_y, args)

    return data_x, data_y, data_stamp


# noinspection DuplicatedCode
class DatasetWind(Dataset):
    """
    Wind dataset for meta-learning
    """
    def __init__(self, args, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT',
                 scale=True,
                 scaler='StandardScaler', timeenc=0, freq='h', lag=0, seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        assert data_path in [f"wind/Zone{i}/Zone{i}.csv" for i in range(1, 11)]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_x = None
        self.data_y = None

        self.features = features
        self.target = target
        self.scale = scale
        if scale:
            if scaler == 'StandardScaler':
                self.scaler = StandardScaler()  # normalize mean to 0, variance to 1
            elif scaler == 'MinMaxScaler':
                self.scaler = MinMaxScaler()  # normalize to [0, 1]
            elif scaler == 'MaxAbsScaler':
                self.scaler = MaxAbsScaler()  # normalize to [-1, 1]
            elif scaler == 'BoxCox':
                self.scaler = PowerTransformer(method='yeo-johnson')  # box-cox transformation
            else:
                raise NotImplementedError
        else:
            self.scaler = None
        self.timeenc = timeenc
        self.freq = freq
        self.lag = lag

        # target path for testing
        # source path for training and evaluation
        self.target_path = os.path.join(root_path, data_path)
        self.source_paths = []
        for i in range(1, 11):
            # if data_path == f"wind/Zone{i}/Zone{i}.csv":
            #     continue
            self.source_paths.append(os.path.join(root_path, f"wind/Zone{i}/Zone{i}.csv"))
        self.__read_data__()

    def __read_data__(self):
        """
        Build meta-learning tasks:
        Each task samples n_way zones, and for each zone selects n_shot support windows
        and n_query query windows.
        """
        if self.set_type == 2:
            # test: single domain, produce windows but still form tasks for evaluation
            zones = [(self.target_path, 0)]
        else:
            # training/validation: multiple source domains
            zones = [(p, i) for i, p in enumerate(self.source_paths)]

        # Collect windowed sequences per zone
        zone_windows = {}  # zone_id -> list of window arrays (seq_len, feat_dim)
        for path, zone_id in zones:
            data_x, data_y, _ = read_data(
                path, self.target, self.seq_len,
                self.set_type, self.features, self.scale,
                self.scaler, self.lag, self.timeenc,
                self.freq, self.args
            )
            # Build sliding windows (forecast input only; use last feature as target label for regression if needed)
            windows = []
            max_start = len(data_x) - self.seq_len - self.pred_len + 1
            for start in range(max_start):
                end = start + self.seq_len
                win_x = data_x[start:end]  # (seq_len, feat_dim)
                windows.append(win_x)
            if len(windows) == 0:
                continue
            zone_windows[zone_id] = np.stack(windows, axis=0)  # (num_windows, seq_len, feat_dim)

        self.tasks = []
        n_way = self.args.n_way
        n_shot = self.args.n_shot
        n_query = self.args.n_query
        n_episode = self.args.n_episode

        # 随机采样n个zone，组成n_way个预测任务
        available_zones = [z for z, arr in zone_windows.items() if arr.shape[0] >= (n_shot + n_query)]
        if len(available_zones) < n_way:
            raise ValueError("Not enough zones with sufficient windows for meta-learning task construction.")

        rng = np.random.default_rng(seed=getattr(self.args, 'seed', 0))

        for _ in range(n_episode):
            # Sample zones for this task
            chosen = rng.choice(available_zones, size=n_way, replace=False)
            x_shot_list, x_query_list, y_shot_list, y_query_list = [], [], [], []
            for class_idx, zone_id in enumerate(chosen):
                arr = zone_windows[zone_id]
                perm = rng.permutation(arr.shape[0])
                support_idx = perm[:n_shot]
                query_idx = perm[n_shot:n_shot + n_query]

                support_windows = arr[support_idx]  # (n_shot, seq_len, feat_dim)
                query_windows = arr[query_idx]  # (n_query, seq_len, feat_dim)

                x_shot_list.append(support_windows)
                x_query_list.append(query_windows)
                y_shot_list.append(np.full(n_shot, class_idx, dtype=np.int64))
                y_query_list.append(np.full(n_query, class_idx, dtype=np.int64))

            x_shot = np.stack(x_shot_list, axis=0)  # (n_way, n_shot, seq_len, feat_dim)
            x_query = np.stack(x_query_list, axis=0)  # (n_way, n_query, seq_len, feat_dim)
            y_shot = np.stack(y_shot_list, axis=0)  # (n_way, n_shot)
            y_query = np.stack(y_query_list, axis=0)  # (n_way, n_query)
            self.tasks.append((x_shot, x_query, y_shot, y_query))

        pass

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        return self.tasks[index]

    # def __read_data__(self):
    #     if self.set_type == 0:
    #         # training set: use source paths
    #         data_x_list = []
    #         data_y_list = []
    #         data_stamp_list = []
    #         for source_path in self.source_paths:
    #             data_x, data_y, data_stamp = read_data(
    #                 source_path, self.target, self.seq_len,
    #                 self.set_type, self.features, self.scale,
    #                 self.scaler, self.lag, self.timeenc,
    #                 self.freq, self.args
    #             )
    #             data_x_list.append(data_x)
    #             data_y_list.append(data_y)
    #             data_stamp_list.append(data_stamp)
    #         self.data_x = np.concatenate(data_x_list, axis=0)
    #         self.data_y = np.concatenate(data_y_list, axis=0)
    #         self.data_stamp = np.concatenate(data_stamp_list, axis=0)
    #     else:
    #         # testing/validation set: use target path
    #         self.data_x, self.data_y, self.data_stamp = read_data(
    #             self.target_path, self.target, self.seq_len,
    #             self.set_type, self.features, self.scale,
    #             self.scaler, self.lag, self.timeenc,
    #             self.freq, self.args
    #         )
    #
    # def __getitem__(self, index):
    #     s_begin = index
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len
    #     r_end = r_begin + self.label_len + self.pred_len
    #
    #     seq_x = self.data_x[s_begin:s_end]
    #     seq_y = self.data_y[r_begin:r_end]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]
    #
    #     return seq_x, seq_y, seq_x_mark, seq_y_mark
    #
    # def __len__(self):
    #     return len(self.data_x) - self.seq_len - self.pred_len + 1
