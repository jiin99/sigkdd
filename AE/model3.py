import torch
import torch.nn.functional as F
from torch import nn, optim
# from dataloader import Firedata_test, Firedata_train
from torch.utils.data import Dataset, DataLoader
import os
import torch.backends.cudnn as cudnn
from torchsummary import summary

# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
       
#         #Encoder-siz
#         self.conv1 = nn.Conv1d(1, 64, kernel_size = 50, padding = 0,stride = 1)  
#         self.conv2 = nn.Conv1d(64, 128,kernel_size = 50, padding = 0,stride = 1)
#         self.conv3 = nn.Conv1d(128, 256,kernel_size = 50, padding = 0,stride = 1)
#         self.conv4 = nn.Conv1d(256, 512,kernel_size = 50, padding = 0,stride = 1)
#         self.conv5 = nn.Conv1d(512, 1024,kernel_size = 50, padding = 0,stride = 1)
#         # self.pool = nn.MaxPool1d(2,stride = 2)
       
#         #Decoder
#         self.t_conv1 = nn.ConvTranspose1d(1024, 512, 50,padding = 0,stride = 1)
#         self.t_conv2 = nn.ConvTranspose1d(512, 256, 50,padding = 0,stride = 1)
#         self.t_conv3 = nn.ConvTranspose1d(256, 128, 50,padding = 0,stride = 1)
#         self.t_conv4 = nn.ConvTranspose1d(128, 64, 50,padding = 0, stride = 1)
#         self.t_conv5 = nn.ConvTranspose1d(64, 1, 50,padding = 0, stride = 1)
#         self.fc = nn.Linear(500,500)



#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = F.relu(self.t_conv1(x))
#         x = F.relu(self.t_conv2(x))
#         x = F.relu(self.t_conv3(x))
#         x = F.relu(self.t_conv4(x))
#         x = F.relu(self.t_conv5(x))
#         return x

# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
       
#         #Encoder-siz
#         self.conv1 = nn.Conv1d(1, 64, kernel_size = 100, padding = 0)  


#         self.t_conv2 = nn.ConvTranspose1d(64, 1, 100,padding = 0)


#     def forward(self, x):
#         x = F.relu(self.conv1(x))

#         x = F.sigmoid(self.t_conv2(x))
              
#         return x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv1d(1, 32, kernel_size = 20, padding = 0)  
        self.conv2 = nn.Conv1d(32, 64,kernel_size = 20, padding = 0)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose1d(64, 32, 20,padding = 0)
        self.t_conv2 = nn.ConvTranspose1d(32, 1, 20,padding = 0)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
              
        return x

class ConvAutoencoder_smooth_cat(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_smooth_cat, self).__init__()
        self.data_module = nn.Sequential(
    
            nn.Conv1d(1, 4, kernel_size = 20, padding = 0),
            nn.ReLU(),
            nn.Conv1d(4, 64, kernel_size = 20, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 4, 20,padding = 0),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, 20,padding = 0),
            )

        self.smooth_module = nn.Sequential(
    
            nn.Conv1d(1, 32, kernel_size = 20, padding = 0),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size = 20, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 20,padding = 0),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 20,padding = 0),
            )
        if torch.cuda.is_available():
            self.data_module = self.data_module.cuda()
            self.smooth_module = self.smooth_module.cuda()


    def forward(self, x):
        out1 = self.data_module(x[:,0,:].unsqueeze(1))
        out2 = self.smooth_module(x[:,1,:].unsqueeze(1))
        
        out = torch.cat((out1,out2),1)
        return out


class ConvAutoencoder_smooth_add(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_smooth_add, self).__init__()
        self.data_module = nn.Sequential(
    
            nn.Conv1d(1, 32, kernel_size = 20, padding = 0),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size = 20, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 20,padding = 0),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 20,padding = 0),
    
            )

        self.smooth_module = nn.Sequential(
    
            nn.Conv1d(1, 32, kernel_size = 20, padding = 0),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size = 20, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 20,padding = 0),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 20,padding = 0),

            )
        if torch.cuda.is_available():
            self.data_module = self.data_module.cuda()
            self.smooth_module = self.smooth_module.cuda()


    def forward(self, x):
        out1 = self.data_module(x[:,0,:].unsqueeze(1))
        out2 = self.smooth_module(x[:,1,:].unsqueeze(1))
        
        out = torch.add(out1,out2)
        import ipdb; ipdb.set_trace()
        return out

class ConvAutoencoder_larger(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_larger, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv1d(1, 64, kernel_size = 20, padding = 0)  
        self.conv2 = nn.Conv1d(64, 128,kernel_size = 20, padding = 0)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose1d(128, 64, 20,padding = 0)
        self.t_conv2 = nn.ConvTranspose1d(64, 1, 20,padding = 0)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
              
        return x

class ConvAutoencoder_smaller(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_smaller, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv1d(1, 16, kernel_size = 20, padding = 0)  
        self.conv2 = nn.Conv1d(16, 32,kernel_size = 20, padding = 0)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose1d(32, 16, 20,padding = 0)
        self.t_conv2 = nn.ConvTranspose1d(16, 1, 20,padding = 0)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
              
        return x

if __name__ == '__main__':
    model = ConvAutoencoder_smaller().cuda()
    summary(model,(1,100))