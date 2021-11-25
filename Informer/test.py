from comet_ml import Experiment
from models.model import Informer, InformerStack
import dataloader
import utils.utils as utils
import time
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
args.save_root = f'./exp-results-informer-len25/'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
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
# Set criterion
# criterion = nn.L1Loss(reduction='sum').cuda()
criterion = nn.MSELoss().cuda()
save_idx = []

def get_global_values(df):
    col = df.columns
    min_value = min(df[col[0]])
    max_value = max(df[col[0]])

    return min_value, max_value

def inverse(x, mini, maxi):
    output = mini + x*(maxi - mini)
    return output

def evaluate(loader, model, criterion, logger,mini,maxi, mode):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    model.eval()
    preds = []
    trues = []
    predictions, loss_list, input_li = np.array([0]),np.array([]),np.array([])
    with torch.no_grad():
            for i, (inputs,target) in enumerate(loader):
                
                inputs = inputs.float().cuda()
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

                loss = criterion(output, target)
                preds = np.array(outputs)
                trues = np.array(trues)
                print('test shape:', preds.shape, trues.shape)
                import ipdb; ipdb.set_trace()
                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                print('test shape:', preds.shape, trues.shape)
                # loss = criterion(output.squeeze(), inputs)
                loss_list = np.append(loss_list,loss.item())
                # input_li.append(inverse(inputs.cpu().data.numpy().squeeze(),mini,maxi).tolist())
                # input_li.append(inverse(inputs.cpu().data.numpy(),mini,maxi).tolist())
                # predictions.append(inverse(output.cpu().data.numpy(),mini,maxi).tolist())
                input_li = np.append(input_li, inputs[:,-1].squeeze().cpu().data.numpy())
                predictions = np.append(predictions,output.squeeze().cpu().data.numpy())

                losses.update(loss.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % args.print_freq == 0:
                    print(mode, ': [{0}/{1}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        i, len(loader), batch_time=batch_time, loss=losses))
            logger.write([losses.avg])

    return input_li, predictions, loss_list, losses.avg

file_list = sorted(os.listdir(args.traindir))
data_list = [file for file in file_list if file.endswith(".csv")]

for i in range(len(data_list)):
    tr_data = pd.read_csv(os.path.join(args.traindir,data_list[i]),index_col = 0)
    min_value, max_value = get_global_values(tr_data)

    test_dataset = dataloader.test_loader(args.traindir,args.testdir,data_list[i])
    save_path = os.path.join(args.save_root,data_list[i], f'trial-{args.trial}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_loader = DataLoader(test_dataset,
                                shuffle=False, 
                                batch_size=args.batch_size, 
                                pin_memory=False)


    test_logger = utils.Logger(os.path.join(save_path, 'test.log'))
    state_dict = torch.load(f'{save_path}/model_50.pth')
    net.load_state_dict(state_dict)
    inputs, pred, in_losses,test_loss = evaluate(test_loader, net, criterion, test_logger,min_value,max_value, 'in')
    idx = [(n,i) for n,i in enumerate(in_losses) if i == max(in_losses)]
    save_idx.append(idx[0][0]*args.seq_len)
    plt.figure(figsize=(64, 16))
    # import ipdb; ipdb.set_trace()
    plt.plot(inputs[idx[0][0]-100 : idx[0][0] + 100], color = 'blue')
    plt.plot(pred[idx[0][0]-100 : idx[0][0] + 100], color = 'red')
    plt.savefig(f'{save_path}/{data_list[i]}1.png')
    plt.close()

    plt.figure(figsize=(64, 16))
    plt.plot(inputs, color = 'blue')
    plt.plot(pred, color = 'red')
    plt.savefig(f'{save_path}/{data_list[i]}_all.png')
    plt.close()

    plt.figure(figsize=(64, 16))
    plt.plot(in_losses[idx[0][0]-1000 : idx[0][0] + 1000])
    plt.savefig(f'{save_path}/{data_list[i]}1_loss.png')
    plt.close()
    import ipdb; ipdb.set_trace()
save_idx_df = pd.DataFrame(save_idx)
save_idx_df.to_csv(f'{args.save_root}/submission1.csv')
    

    # with open(f'{save_path}/idx.csv', 'a', newline='') as f:
    #         columns = ["",
    #                    "LOSS",
    #                    "AUROC",
    #                    "FPR@95%TPR",
    #                    "AUPRC"]
    #         writer = csv.writer(f)
    #         writer.writerow(['fire early detection'])
    #         writer.writerow(columns)
    #         writer.writerow(
    #             ['', test_loss,
    #              100 * auroc,
    #              100 * fpr,
    #              100 * auprc])
    #         writer.writerow([''])
    # f.close()

# for i in range(32):
#     plt.plot(in_losses[i*10:i*10+10],label = 'in')
#     plt.plot(out_losses[i*10:i*10+10],label = 'out')
#     plt.legend()
#     plt.savefig(f'./exp-results-hidden{args.hidden_size}/trial-{args.trial}/{i}th_test_sub_20')
#     plt.close()