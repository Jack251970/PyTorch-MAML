import os
from turtle import pd

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer
from torch.utils.data import Dataset

from datasets.uea import interpolate_missing
from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features


# noinspection DuplicatedCode
class DatasetWind(Dataset):
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

        self.path = os.path.join(root_path, data_path)
        self.__read_data__()

    def __read_data__(self):
        # read raw data
        df_raw = pd.read_csv(self.path)

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
        cols.remove(self.target)
        cols.remove(date_column)

        # reorganize df_raw
        df_raw = df_raw[[date_column] + cols + [self.target]]

        # divide data into train, vali, test parts
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # get boarders of the data
        # set_type: {'train': 0, 'val': 1, 'test': 2}
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # select features
        if self.features == 'M' or self.features == 'MS':
            # select features except the first
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # select features only the target
            df_data = df_raw[[self.target]]
        else:
            raise NotImplementedError

        # apply standard scaler if needed
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # add lag feature
        if self.lag > 0:
            for i in range(self.lag, 0, -1):
                label_data = data[:, -1]  # {ndarray: (N,)}
                lag_data = np.concatenate([np.zeros(i), label_data[:-i]], axis=0)  # {ndarray: (N,)}
                data = np.concatenate([lag_data.reshape(-1, 1), data], axis=1)

        # extract date column
        df_stamp = df_raw[[date_column]][border1:border2]
        df_stamp[date_column] = pd.to_datetime(df_stamp.date)

        # encode time
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([date_column], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[date_column].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError('timeenc should be 0 or 1!')

        # output data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_scaler(self):
        return self.scaler

    def get_all_data(self):
        return self.data_x, self.data_y, self.data_stamp
