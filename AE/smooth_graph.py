import time
import dataloader
import model
import model2
import model3
import utils
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
parser = argparse.ArgumentParser(description='Fire detection')

parser.add_argument('--model', default='auto3', type=str, help='epoch (default: 50)')
parser.add_argument('--epochs', default=50, type=int, help='epoch (default: 200)')
parser.add_argument('--batch-size', default=1024, type=int, help='batch size (default: 128)')
# parser.add_argument('--save-root', default='./exp-results-hidden64/', type=str, help='save root')
parser.add_argument('--hidden-size', default=128, type=int, help='hidden size (default: 128)')
parser.add_argument('--seq-len', default=100, type=int, help='hidden size (default: 128)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--trial', default='01', type=str)
parser.add_argument('--traindir',default = '/daintlab/data/sigkdd2021/PhaseII/trainset', type=str, help='train data path')
parser.add_argument('--testdir',default = '/daintlab/data/sigkdd2021/PhaseII/testset', type=str, help='test data path')
parser.add_argument('--gpu-id', default='1', type=str, help='gpu number')

args = parser.parse_args()
args.save_root = f'./exp-results-len100-hidden32-model3-aug2/'


# print(test_point)
file_list = sorted(os.listdir(args.testdir))
data_list = [file for file in file_list if file.endswith(".csv")]
divide = 500

for i in range(1,250):
    data = pd.read_csv(os.path.join(args.save_root,data_list[i],'trial-01','output.csv'),index_col = 0)
    save_path = os.path.join(args.save_root,data_list[i], f'trial-{args.trial}')
    idx = data['loss'].idxmax()     
    
    for k in range(int(data.shape[0]/divide)):
        print(k)
        plt.figure(figsize=(64, 16))
        plt.plot(data['data'][k*divide: k*divide + divide], color = 'blue', label = 'data', linewidth=3)
        plt.plot(data['pred'][k*divide: k*divide + divide], color = 'red', label = 'prediction', linewidth=3)
        plt.legend(loc='upper right',fontsize = 40)
        if idx in range(k*divide,k*divide+divide):
            plt.axvline(x=idx - (k*divide), color='black', linestyle='--', linewidth=3)
            plt.savefig(f'{save_path}/recons_max_loss.png')
        plt.savefig(f'{save_path}/recons_{k}.png')
        plt.close()

    # for k in range(int(len(inputs)/divide)):
    #     for num,p in enumerate(top20idx):
    #         if p in range(k*divide,k*divide+divide):
    #             plt.figure(figsize=(64, 16))
    #             plt.plot(inputs[k*divide: k*divide + divide], color = 'blue', label = 'data', linewidth=3)
    #             plt.plot(pred[k*divide: k*divide + divide], color = 'red', label = 'prediction', linewidth=3)
    #             plt.legend(loc='upper right',fontsize = 40)
    #             if idx[0][0]+50 in range(k*divide,k*divide+divide):
    #                 plt.axvline(x=idx[0][0]+ 50 - (k*divide), color='black', linestyle='--', linewidth=3)
    #                 plt.savefig(f'{save_path}/recons_max_loss.png')
    #             plt.savefig(f'{save_path}/top_{num}.png')
    #             plt.close()
    #         else :
    #             plt.figure(figsize=(64, 16))
    #             plt.plot(inputs[k*divide: k*divide + divide], color = 'blue', label = 'data', linewidth=3)
    #             plt.plot(pred[k*divide: k*divide + divide], color = 'red', label = 'prediction', linewidth=3)
    #             plt.legend(loc='upper right',fontsize = 40)
    #             if idx[0][0]+50 in range(k*divide,k*divide+divide):
    #                 plt.axvline(x=idx[0][0]+ 50 - (k*divide), color='black', linestyle='--', linewidth=3)
    #                 plt.savefig(f'{save_path}/recons_max_loss1.png')
    #             plt.savefig(f'{save_path}/recons_{k}.png')
    #             plt.close()
    #     plt.savefig(f'{save_path}/top_{num}.png')
    #     plt.close()
    # plt.plot(inputs[idx[0][0] + test_point-200 : idx[0][0] + test_point+ 200], color = 'blue', label = 'data')
    # plt.plot(pred[idx[0][0] + test_point-200 : idx[0][0] + test_point + 200], color = 'red', label = 'prediction')
    # plt.legend(loc='upper right',fontsize = 40)
    # plt.savefig(f'{save_path}/{data_list[i]}2.png')
    # plt.close()


    plt.figure(figsize=(64, 16))
    plt.plot(data['data'], color = 'blue',linewidth = 10, alpha = 0.5, label = 'data')
    plt.plot(data['pred'], color = 'red', linewidth = 10, alpha = 0.5,label = 'prediction')
    plt.legend(loc='upper right',fontsize = 40)
    plt.savefig(f'{save_path}/{data_list[i]}_all.png')
    plt.close()


    # for i in range(32):
    #     plt.plot(in_losses[i*10:i*10+10],label = 'in')
    #     plt.plot(out_losses[i*10:i*10+10],label = 'out')
    #     plt.legend()
    #     plt.savefig(f'./exp-results-hidden{args.hidden_size}/trial-{args.trial}/{i}th_test_sub_20')
    #     plt.close()