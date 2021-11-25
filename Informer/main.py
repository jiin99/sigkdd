from comet_ml import Experiment
from models.model import Informer, InformerStack
import dataloader
import utils.utils as utils
from utils.tools import EarlyStopping, adjust_learning_rate
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['agg.path.chunksize'] = 10000
parser = argparse.ArgumentParser(description='Fire detection')

parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--epochs', default=200, type=int, help='epoch (default: 200)')
parser.add_argument('--batch-size', default=2048, type=int, help='batch size (default: 1024)')
# parser.add_argument('--save-root', default='./exp-results-hidden32/', type=str, help='save root')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--hidden-size', default=32, type=int, help='hidden size (default: 128)')
parser.add_argument('--trial', default='01', type=str)
parser.add_argument('--traindir',default = '/daintlab/data/sigkdd2021/PhaseII/trainset', type=str, help='train data path')
parser.add_argument('--testdir',default = '/daintlab/data/sigkdd2021/PhaseII/testset', type=str, help='test data path')
parser.add_argument('--gpu-id', default='3', type=str, help='gpu number')


parser.add_argument('--seq_len', type=int, default=100, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=50, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=25, help='prediction sequence length')

parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.save_root = f'./exp-results-informer-len100/'
# len100 : [100, 99,1 ]
# len25 : [100,50,25 ]
def train(loader, net, criterion, optimizer, epoch, logger,experiment, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    net.train()
    for i, (input,target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().cuda()
        target = target.float().cuda()
        
        if args.padding==0:
            dec_inp = torch.zeros([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
        elif args.padding==1:
            dec_inp = torch.ones([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
        
        dec_inp = torch.cat([target[:,:args.label_len,:], dec_inp], dim=1).float().cuda()
        
        if args.output_attention:
            outputs = net(input,dec_inp)[0]
        else:
            outputs, attns = net(input, dec_inp)
        
        target = target[:,-args.pred_len:,0:].cuda()
        # print(i,'=======', outputs.shape, target.shape)
        loss = criterion(outputs, target)
        losses.update(loss.item(), input.size(0))

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
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

    # adjust_learning_rate(optimizer, epoch+1, args)
    logger.write([epoch, losses.avg])
    experiment.log_metric("loss", losses.avg, epoch = epoch)
   
    return epoch, losses.avg

def test(loader, net, criterion, epoch, logger,experiment, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    net.eval()
    for i, (input,target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().cuda()
        target = target.float().cuda()
        
        if args.padding==0:
            dec_inp = torch.zeros([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
        elif args.padding==1:
            dec_inp = torch.ones([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
        
        dec_inp = torch.cat([target[:,:args.label_len,:], dec_inp], dim=1).float().cuda()
        
        if args.output_attention:
            outputs, attn = net(input,dec_inp)[0]
        else:
            outputs, attns = net(input, dec_inp)
        
        target = target[:,-args.pred_len:,0:]
        # print(i,'=======', outputs.shape, target.shape)
        # import ipdb; ipdb.set_trace()
        loss = criterion(outputs.detach().cpu(), target.detach().cpu())
        
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

    # adjust_learning_rate(optimizer, epoch+1, args)
    logger.write([epoch, losses.avg])
    # experiment.log_metric("loss", losses.avg, epoch = epoch)
   
    return epoch, losses.avg


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    file_list = sorted(os.listdir(args.traindir))
    data_list = [file for file in file_list if file.endswith(".csv")]
    
    #225부터 시작.
    for i in range(180,250):
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

        train_dataset = dataloader.train_loader(args.traindir,data_list[i])
                                                                            # ,transforms.Compose([Jittering(0.1),Scaling(0.1),Rotation()]))

        test_dataset = dataloader.test_loader(args.traindir,args.testdir,data_list[i])

        save_path = os.path.join(args.save_root,data_list[i], f'trial-{args.trial}')
        args.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_loader = DataLoader(train_dataset,
                                    shuffle=False, 
                                    batch_size=args.batch_size, 
                                    pin_memory=False)
        test_loader = DataLoader(test_dataset,
                                shuffle=False, 
                                batch_size=args.batch_size, 
                                pin_memory=False)


        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if args.model=='informer' or args.model=='informerstack':
            e_layers = args.e_layers if args.model=='informer' else args.s_layers
            net = model_dict[args.model](
                args.enc_in,
                args.dec_in,
                args.c_out, 
                args.seq_len, 
                args.label_len,
                args.pred_len, 
                args.factor,
                args.d_model, 
                args.n_heads, 
                e_layers, # self.args.e_layers,
                args.d_layers, 
                args.d_ff,
                args.dropout, 
                args.attn,
                args.embed,
                args.activation,
                args.output_attention,
                args.distil,
                args.mix
            ).float().cuda()
        print(net)
        import ipdb; ipdb.set_trace()
        # net = nn.DataParallel(net)

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
                epoch,loss = train(train_loader,
                                    net,
                                    criterion,
                                    optimizer,
                                    epoch,
                                    train_logger,
                                    experiment,
                                    args)
                epoch, tst_loss = test(test_loader, net, criterion, epoch, test_logger,experiment, args)
        
        torch.save(net.state_dict(),
                os.path.join(save_path, f'model_{int(args.epochs)}.pth'))

if __name__ == "__main__":
    main()