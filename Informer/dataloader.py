import numpy as np
import pandas as pd
import os, glob
import os.path
import torch
import random
from random import randrange
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
# from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
def get_data(targetdir,type_,file_name):
    df = pd.read_csv(os.path.join(targetdir,type_,file_name))
    return df

def get_global_values(df):
    col = df.columns
    min_value = min(df[col[0]])
    max_value = max(df[col[0]])
    # import ipdb; ipdb.set_trace()
    return min_value, max_value

def minmax_scaling(x,min_value,max_value):
    x[:,0] = (x[:,0] - min_value)/(max_value - min_value)
    
    return x

class train_loader(Dataset):
    def __init__(self, targetdir,file_name,seq_size = [100, 50, 25],transform = None):
        data = pd.read_csv(os.path.join(targetdir,file_name),index_col = 0)
        min_value, max_value = get_global_values(data)
        self.data = minmax_scaling(data.values,min_value,max_value)
        self.seq_len = seq_size[0]
        self.label_len = seq_size[1]
        self.pred_len = seq_size[2]
        self.transform = transform
        indices = []
        #stride 10
        for i in range(len(self.data)):
            indices.append(i)
        self.indices = indices[:-(self.seq_len+self.pred_len) + 1]

    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, index):
        
        s_begin = self.indices[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        x = self.data[s_begin:s_end, 0].reshape(-1,1)
        y = self.data[r_begin : r_end,0].reshape(-1,1)

        # idx = self.indices[index]
        # x = self.data[s_begin:s_end, 0].reshape(-1,1)
        # y = self.data[s_begin + 1 : s_end + 1,0].reshape(-1,1)

        if self.transform:
            x = self.transform(x)
        
        return x, y

class train_loader_smooth(Dataset):
    def __init__(self, targetdir,file_name,seq_size = [100, 99, 1],transform = None):
        data = pd.read_csv(os.path.join(targetdir,file_name),index_col = 0)
        # data_em = data.rolling(20,min_periods=1,center=True).mean()
        min_value, max_value = get_global_values(data)
        self.data = minmax_scaling(data.rolling(30,min_periods=1,center=True).mean().values,min_value,max_value)
        self.seq_len = seq_len
        self.transform = transform
        indices = []
        #stride 10
        for i in range(len(self.data)):
            indices.append(i)
        self.indices = indices[:-(self.seq_len)]

    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, index):
        
        idx = self.indices[index]
        x = self.data[idx : idx + self.seq_len, 0].reshape(1,-1)
        y = self.data[idx + self.seq_len].reshape(1,-1)
        if self.transform:
            x = self.transform(x)
        
        return x, y

class test_loader(Dataset):
    def __init__(self, traindir, testdir,file_name,seq_size = [100, 50, 25],transform = None):
        train_data = pd.read_csv(os.path.join(traindir,file_name),index_col = 0)
        min_value, max_value = get_global_values(train_data)
        data = pd.read_csv(os.path.join(testdir,file_name),index_col = 0)
        # import ipdb; ipdb.set_trace()
        self.data = minmax_scaling(data.values,min_value,max_value)
        self.seq_len = seq_size[0]
        self.label_len = seq_size[1]
        self.pred_len = seq_size[2]
        self.transform = transform
        indices = []
        #stride 10
        for i in range(len(self.data)):
            indices.append(i)
        self.indices = indices[:-(self.seq_len+self.pred_len) + 1]

    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, index):
        
        s_begin = self.indices[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        x = self.data[s_begin:s_end, 0].reshape(-1,1)
        y = self.data[r_begin : r_end,0].reshape(-1,1)

        # idx = self.indices[index]
        # x = self.data[s_begin:s_end, 0].reshape(-1,1)
        # y = self.data[s_begin + 1 : s_end + 1,0].reshape(-1,1)

        if self.transform:
            x = self.transform(x)
        
        return x, y

class test_loader_smooth(Dataset):
    def __init__(self, traindir, testdir,file_name,seq_len):
        train_data = pd.read_csv(os.path.join(traindir,file_name),index_col = 0)
        min_value, max_value = get_global_values(train_data)
        data = pd.read_csv(os.path.join(testdir,file_name),index_col = 0)
        self.data = minmax_scaling(data.rolling(30,min_periods=1,center=True).mean().values,min_value,max_value)
        self.seq_len = seq_len
        indices = []
        # 오른쪽 두개 오백
        for i in range(len(self.data)-self.seq_len):
            indices.append(i)
        
        self.indices = indices

    def __len__(self):
        # return len(self.data)-10
        return len(self.indices)
        
    def __getitem__(self, index):
        # idx = index
        idx = self.indices[index]
        x = self.data[idx : idx + self.seq_len, 0].reshape(1,-1)
        
        return x, idx


class Jittering(object):
    def __init__(self, sigma):
        assert isinstance(sigma, (float, tuple))
        self.sigma = sigma

    def __call__(self, sample):
        #print(sample)
        data, label = sample['data'], sample['label']
        if isinstance(self.sigma, float):
            myNoise = np.random.normal(loc=0, scale=self.sigma, size=data.shape)
            data = data+myNoise
            factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
            data = np.multiply(data, factor[:,np.newaxis,:])
            flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
            rotate_axis = np.arange(x.shape[2])
            np.random.shuffle(rotate_axis)    
            data = flip[:,np.newaxis,:] * data[:,:,rotate_axis]
        return {'data': data, 'label': label}

class Jittering(object):
    def __init__(self, sigma):
        assert isinstance(sigma, (float, tuple))
        self.sigma = sigma

    def __call__(self, sample):
        
        data = sample
        if isinstance(self.sigma, float):
            
            myNoise = np.random.normal(loc=0, scale=self.sigma, size=data.shape)
            data = data+myNoise

        return data

class Scaling(object):
    def __init__(self, sigma):
        assert isinstance(sigma, (float, tuple))
        self.sigma = sigma

    def __call__(self, sample):
        
        data = sample
        if isinstance(self.sigma, float):
            
            factor = np.random.normal(loc=1., scale=self.sigma, size=data.shape)
            data = np.multiply(data, factor[:,:])

        return data

class Rotation(object):
    
    def __call__(self, sample):
        
        data = sample
        flip = np.random.choice([-1, 1])
        data = flip * data
        
        return data


if __name__ == "__main__":
    traindir = r'/daintlab/data/sigkdd2021/PhaseII/trainset'
    testdir = r'/daintlab/data/sigkdd2021/PhaseII/testset'

    file_list = sorted(os.listdir(testdir))
    data_list = [file for file in file_list if file.endswith(".csv")]

    tr_dataset = train_loader_smooth(traindir,data_list[249],100)
    tr_dl = DataLoader(tr_dataset, shuffle=False, batch_size=10, pin_memory=False)

    for i in range(len(tr_dl)):
        x,y = next(iter(tr_dl))
        
 
