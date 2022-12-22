import os
import numpy as np
import pandas as pd
import pickle
from imblearn.under_sampling import RandomUnderSampler

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_BTC2(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='MS', data_path='ETTm1.csv',
                 use_decoder_tokens=False, date_period1=None, date_period2=None,date_period3=None,
                 target='cl', scale=True, timeenc=0, freq='t', option=0):

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_decoder_tokens = use_decoder_tokens

        self.root_path = root_path
        self.data_path = data_path
        self.date_period1 = date_period1
        self.date_period2 = date_period2
        self.date_period3 = date_period3
        self.option = option
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_target = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if "Unnamed: 0" in df_raw.columns:
            df_raw = df_raw.drop(columns="Unnamed: 0")

        range1 = 0
        range2 = len(df_raw)

        if self.set_type == 0:
            df_raw = df_raw[(df_raw['date'] >= self.date_period1)&(df_raw['date'] < self.date_period2)]
        else:
            df_raw = df_raw[(df_raw['date'] >= self.date_period2) & (df_raw['date'] < self.date_period3)]

        df_raw = df_raw.reset_index(drop=True)
        train_columns = ['date', 'op', 'hi', 'lo', 'cl', 'volume']
        if 'pred' in self.data_path:
            df_target = df_raw[['target1', 'target2', 'target3']]
        else:
            target_column = 'target'
            df_target = pd.get_dummies(df_raw[target_column])

        df_raw = df_raw[train_columns]

        border1s = [range1, range1, range1]
        border2s = [range2, range2, range2]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_raw['spread'] = df_raw['hi'] - df_raw['lo']
        df_raw['transition'] = df_raw['cl'] - df_raw['op']
        df_raw['tran_rate'] = abs(df_raw['cl'] - df_raw['op'])*100 / df_raw['spread']
        df_raw['tran_rate'] = df_raw['tran_rate'].fillna(0)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            if self.set_type == 0:
                self.scaler = StandardScaler()
                self.scaler.fit(train_data.values)
                pickle.dump(self.scaler, open("scaler.pkl", "wb"))

            else:
                self.scaler = pickle.load(open('scaler.pkl', 'rb'))
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        target_data = df_target[border1s[0]:border2s[0]]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)

        data_x = data[border1:border2]
        data_y = target_data.values[border1:border2]

        self.data_x = data_x
        self.data_y = data_y
        self.data_stamp = data_stamp
        df_raw['date'] = df_raw['date'].apply(lambda x: int(x[:4] + x[5:7] + x[8:10] + x[11:13] + x[14:16]))
        self.data_val = df_raw[['date', 'op', 'hi', 'lo', 'cl']].values[border1:border2]



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        if not self.use_decoder_tokens:
            # decoder without tokens
            r_begin = s_end
            r_end = r_begin + self.pred_len

        else:
            # decoder with tokens
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_val = self.data_val[r_begin:r_end]

        if  (self.set_type == 1) or (self.set_type == 2):
            return index, seq_x, seq_y, seq_x_mark, seq_y_mark, seq_val
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_val

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_BTC_pred:
    def __init__(self, root_path, flag='train', size=None, features='MS', data_path='ETTm1.csv',
                 use_decoder_tokens=False, date_period1=None, date_period2=None,
                 target='cl', scale=True, timeenc=0, freq='t', mode=1):

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_decoder_tokens = use_decoder_tokens

        self.root_path = root_path
        self.data_path = data_path
        self.date_period1 = date_period1
        self.date_period2 = date_period2

        self.scaler = pickle.load(open('scaler.pkl', 'rb'))
        self.scaler_target = pickle.load(open('scaler_target.pkl', 'rb'))

        self.mode = mode

    def read_data(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if "Unnamed: 0" in df_raw.columns:
            df_raw = df_raw.drop(columns="Unnamed: 0")

        df_raw = df_raw[(df_raw['date'] >= self.date_period1) & (df_raw['date'] < self.date_period2)]
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_data['spread'] = df_data['hi'] - df_data['lo']
        df_data['transition'] = df_data['cl'] - df_data['op']
        df_raw['tran_rate'] = abs(df_raw['cl'] - df_raw['op']) * 100 / df_raw['spread']
        df_raw['tran_rate'] = df_raw['tran_rate'].fillna(0)

        df_target = (df_data['hi'] + df_data['lo']) / 2

        data = self.scaler.transform(df_data.values)
        data_y = self.scaler_target.transform(df_target.values)

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)

        data_x = data
        data_y = np.expand_dims(data_y, 1)
        df_raw['date'] = df_raw['date'].apply(lambda x: int(x[:4] + x[5:7] + x[8:10] + x[11:13] + x[14:16]))
        data_val = df_raw[['date', 'op', 'hi', 'lo', 'cl']].values
        return data_x, data_y, data_stamp, data_val

    def extract_data(self, data_values, target_val, data_stamp, df_raw):
        data_len = len(data_values) - self.seq_len
        if self.mode == 1:
            for index in range(0, data_len, self.pred_len):
                s_begin = index
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = data_values[s_begin:s_end]
                seq_y = target_val[r_begin:r_end]
                seq_raw = df_raw[r_begin:r_end]
                yield index, seq_x, seq_y, seq_raw
        else:
            for index in range(0, data_len):
                s_begin = index
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = data_values[s_begin:s_end]
                seq_y = target_val[r_begin:r_end]
                seq_raw = df_raw[r_begin:r_end]
                yield index, seq_x, seq_y, seq_raw

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
