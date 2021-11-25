from comet_ml import Experiment

import dataloader
import model
import model2
import model3
import utils
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from dataloader import Jittering, Scaling, Rotation
plt.rcParams['agg.path.chunksize'] = 10000
parser = argparse.ArgumentParser(description='Fire detection')
parser.add_argument('--epochs', default=200, type=int, help='epoch (default: 200)')
parser.add_argument('--batch-size', default=1024, type=int, help='batch size (default: 128)')
# parser.add_argument('--save-root', default='./exp-results-hidden32/', type=str, help='save root')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--hidden-size', default=32, type=int, help='hidden size (default: 128)')
parser.add_argument('--seq-len', default=100, type=int, help='hidden size (default: 128)')
parser.add_argument('--trial', default='01', type=str)
parser.add_argument('--traindir',default = '/daintlab/data/sigkdd2021/PhaseII/trainset', type=str, help='train data path')
parser.add_argument('--testdir',default = '/daintlab/data/sigkdd2021/PhaseII/testset', type=str, help='test data path')
parser.add_argument('--gpu-id', default='3', type=str, help='gpu number')

args = parser.parse_args()

args.save_root = f'./exp-results-len{args.seq_len}-hidden{args.hidden_size}-model3-smooth-add/'

def train(raw_loader,smooth_loader, net, criterion, optimizer, epoch, logger,experiment, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    net.train()
    for i, (raw,smooth) in enumerate(zip(raw_loader,smooth_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        input = raw[0].float().cuda()
        target = raw[1].float().cuda()
        smooth_input = smooth[0].float().cuda()
        smooth_target = smooth[1].float().cuda()
        
        
        data = torch.cat((input,smooth_input),1).cuda()
              
        output = net(data)
        #cat
        # loss = criterion(output, data)
        #add
        import ipdb; ipdb.set_trace()
        loss = criterion(output, data)
        losses.update(loss.item(), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(raw_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

    logger.write([epoch, losses.avg])
    experiment.log_metric("loss", losses.avg, epoch = epoch)
  
    return epoch, losses.avg


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    file_list = sorted(os.listdir(args.traindir))
    data_list = [file for file in file_list if file.endswith(".csv")]
    
    #225부터 시작.
    for i in range(250):
        hyper_params = {
            "batch_size" : args.batch_size,
            "num_epochs" : args.epochs,
            "trial" : args.trial,
            "save_root" : args.save_root,
            "dataset" : data_list[i]}

        experiment = Experiment(
            api_key="2KS4Pd7VAw7wcFh5ZkDDhshm2",
            project_name="sigkdd2021",
            workspace="jiin99", disabled = True
        )
        experiment.log_parameters(hyper_params)

        train_dataset = dataloader.train_loader(args.traindir,data_list[i],args.seq_len)
        train_dataset_smooth = dataloader.train_loader_smooth(args.traindir,data_list[i],args.seq_len)

        save_path = os.path.join(args.save_root,data_list[i], f'trial-{args.trial}')
        args.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_loader = DataLoader(train_dataset,
                                    shuffle=False, 
                                    batch_size=args.batch_size, 
                                    pin_memory=False)
        train_loader_smooth = DataLoader(train_dataset_smooth,
                                    shuffle=False, 
                                    batch_size=args.batch_size, 
                                    pin_memory=False)

        embedding_dim = args.hidden_size
        # num_layers = 1
 
        # net = model.LstmAutoEncoder(args.seq_len, 1, embedding_dim,num_layers)
        # net = model2.RecurrentAutoencoder(args.seq_len, 1, embedding_dim).cuda()
        net = model3.ConvAutoencoder_smooth_cat().cuda()
        # net = model3.ConvAutoencoder_smooth_add().cuda()

        criterion = nn.MSELoss().cuda()

        optimizer = optim.Adam(net.parameters(),lr=1e-3)
        scheduler = MultiStepLR(optimizer,
                                milestones=[100,150],
                                gamma=0.1)
        train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
        test_logger = utils.Logger(os.path.join(save_path, 'test.log'))

        # Start Train
        for epoch in range(1, args.epochs+1):
            with experiment.train() : 
                epoch,loss = train(train_loader,train_loader_smooth,
                                    net,
                                    criterion,
                                    optimizer,
                                    epoch,
                                    train_logger,
                                    experiment,
                                    args)
            scheduler.step()
        
        torch.save(net.state_dict(),
                os.path.join(save_path, f'model_{int(args.epochs)}.pth'))

if __name__ == "__main__":
    main()