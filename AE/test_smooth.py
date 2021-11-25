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
args.save_root = f'./exp-results-len100-hidden32-model3-smooth-add/'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
embedding_dim = args.hidden_size
num_layers = 1
# net = model.LstmAutoEncoder(10, 2, embedding_dim, num_layers).cuda()
if args.model == 'auto1' : 
    net = model.LstmAutoEncoder(args.seq_len, 1, embedding_dim, num_layers).cuda()
elif args.model == 'auto2' : 
    net = model2.RecurrentAutoencoder(args.seq_len, 1, embedding_dim).cuda()
elif args.model == 'auto3' : 
    net = model3.ConvAutoencoder_smooth_add().cuda()
elif args.model == 'lstm':
    net = lstm.LSTM(1,50,100).cuda()

# Set criterion
# criterion = nn.L1Loss(reduction='sum').cuda()
criterion = nn.MSELoss(size_average = False,reduce = False).cuda()
save_idx = []

def get_global_values(df):
    col = df.columns
    min_value = min(df[col[0]])
    max_value = max(df[col[0]])

    return min_value, max_value

def inverse(x, mini, maxi):
    output = mini + x*(maxi - mini)
    return output

def evaluate(raw_loader,smooth_loader, model, criterion, logger,mini,maxi, length):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    model.eval()
    loss_list, input_li, smooth_input_li = np.array([]),np.array([]),np.array([])
    # agg_dic = dict.fromkeys(range(length+100),[])
    agg_dic = dict(zip(range(length+99),[[]for i in range(length+99)]))
    agg_dic_smooth = dict(zip(range(length+99),[[]for i in range(length+99)]))
    with torch.no_grad():
            for i, (raw,smooth) in enumerate(zip(raw_loader,smooth_loader)):
                input = raw[0].float().cuda()
                smooth_input = smooth[0].float().cuda()
                data = torch.cat((input,smooth_input),1).cuda()
                
                output = model(data)
                import ipdb; ipdb.set_trace()

                out_li = output.squeeze().cpu().data.tolist()
                # smooth_out_li = output[:,1,:].cpu().data.tolist()
                idx_li = raw[1].cpu().data.tolist()
                # import ipdb; ipdb.set_trace()
                [agg_dic[idx_li[i]+j].append(out_li[i][j]) for i in range(len(input)) for j in range(len(out_li[i]))]
                # [agg_dic_smooth[idx_li[i]+j].append(smooth_out_li[i][j]) for i in range(len(smooth_input)) for j in range(len(smooth_out_li[i]))]
                
                # loss = criterion(output, inputs)
                # loss = criterion(output.squeeze(), inputs)
                # loss_list = np.append(loss_list,[sum(k).item() for k in loss.squeeze()])
                
                # input_li.append(inverse(inputs.cpu().data.numpy().squeeze(),mini,maxi).tolist())
                # input_li.append(inverse(inputs.cpu().data.numpy(),mini,maxi).tolist())
                # predictions.append(inverse(output.cpu().data.numpy(),mini,maxi).tolist())
                if i == 0 : 
                    input_li = np.append(input_li, input.squeeze()[0].cpu().data.numpy())
                    input_li = np.append(input_li, input.squeeze()[1:,-1].cpu().data.numpy())
                    smooth_input_li = np.append(smooth_input_li, smooth_input.squeeze()[0].cpu().data.numpy())
                    smooth_input_li = np.append(smooth_input_li, smooth_input.squeeze()[1:,-1].cpu().data.numpy())
                else : 
                    input_li = np.append(input_li, input.squeeze()[:,-1].cpu().data.numpy())
                    smooth_input_li = np.append(smooth_input_li, smooth_input.squeeze()[:,-1].cpu().data.numpy())

                # losses.update(loss.item(), inputs.size(0))
                # import ipdb; ipdb.set_trace()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % args.print_freq == 0:
                    print(': [{0}/{1}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        i, len(raw_loader), batch_time=batch_time, loss=losses))

            predictions = [np.mean(agg_dic[i]) for i in agg_dic.keys()]  
            # smooth_predictions = [np.mean(agg_dic_smooth[i]) for i in agg_dic_smooth.keys()]  
            
            logger.write([losses.avg])
            

    return input_li, predictions, smooth_input_li, losses.avg

file_list = sorted(os.listdir(args.testdir))
data_list = [file for file in file_list if file.endswith(".csv")]
# print(len(data_list))

for i in range(7,250):
    tr_data = pd.read_csv(os.path.join(args.traindir,data_list[i]),index_col = 0)
    min_value, max_value = get_global_values(tr_data)

    test_dataset = dataloader.test_loader(args.traindir,args.testdir,data_list[i],args.seq_len)
    test_dataset_smooth = dataloader.test_loader_smooth(args.traindir,args.testdir,data_list[i],args.seq_len)
    
    save_path = os.path.join(args.save_root,data_list[i], f'trial-{args.trial}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_loader = DataLoader(test_dataset,
                                shuffle=False, 
                                batch_size=args.batch_size, 
                                pin_memory=False)
    test_loader_smooth = DataLoader(test_dataset_smooth,
                                shuffle=False, 
                                batch_size=args.batch_size, 
                                pin_memory=False)
    
    test_logger = utils.Logger(os.path.join(save_path, 'test.log'))
    idx_logger = utils.Logger(os.path.join(save_path, 'idx1.log'))
    idx_logger_global = utils.Logger(os.path.join(args.save_root, 'idx1_250.log'))
    state_dict = torch.load(f'{save_path}/model_200.pth')
    net.load_state_dict(state_dict)
    
    inputs, pred, smooth_inputs,test_loss = evaluate(test_loader,test_loader_smooth, net, criterion, test_logger,min_value,max_value, len(test_dataset))
    
    in_losses = criterion(torch.add(torch.tensor(inputs).cuda(),torch.tensor(smooth_inputs).cuda()),torch.tensor(pred).cuda())
    # smooth_in_losses = criterion(torch.tensor(smooth_inputs).cuda(),torch.tensor(smooth_pred).cuda())
    # total_losses = torch.add(in_losses,smooth_in_losses)
    
    idx = torch.argmax(in_losses).cpu().data.numpy()
    import ipdb; ipdb.set_trace()
    # in_losses_npy =in_losses.cpu().data.numpy()
    save_output = pd.DataFrame({'data' : inputs+smooth_inputs,'pred' :pred})
    # save_output['loss'] = 0
    save_output['loss'] = in_losses.cpu().data.numpy()
    
    save_output.to_csv(f'{save_path}/output.csv')
    
    test_point = data_list[i].rstrip('.csv')
    test_point = int(test_point.split('_')[-1])
    # print(test_point)
    save_idx.append(idx + test_point)
    idx_logger.write([int(idx + test_point)])
    idx_logger_global.write([int(idx + test_point)])
    print(data_list[i])
    # print(idx[0][0] + test_point)

    # top20= sorted(range(len(in_losses)),key= lambda i: in_losses[i])[-20:]
    # top20idx = [i+50 for i in top20]

    # divide = 500
    
    # for k in range(int(len(inputs)/divide)):
    #     print(k)
    #     plt.figure(figsize=(64, 16))
    #     plt.plot(inputs[k*divide: k*divide + divide], color = 'blue', label = 'data', linewidth=3)
    #     plt.plot(pred[k*divide: k*divide + divide], color = 'red', label = 'prediction', linewidth=3)
    #     plt.legend(loc='upper right',fontsize = 40)
    #     if idx[0][0]+50 in range(k*divide,k*divide+divide):
    #         plt.axvline(x=idx[0][0]+ 50 - (k*divide), color='black', linestyle='--', linewidth=3)
    #         plt.savefig(f'{save_path}/recons_max_loss.png')
    #     plt.savefig(f'{save_path}/recons_{k}.png')
    #     plt.close()


save_idx_df = pd.DataFrame(save_idx)
save_idx_df.reset_index(inplace = True)
save_idx_df.columns = ['No.','location']
save_idx_df['No.'] = save_idx_df['No.'] + 1
save_idx_df.set_index('No.')
save_idx_df.to_csv(f'{args.save_root}/submission_250.csv')
    

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