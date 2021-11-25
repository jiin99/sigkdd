import torch
import torch.nn.functional as F
from torch import nn, optim
# from dataloader import Firedata_test, Firedata_train
from torch.utils.data import Dataset, DataLoader
import os
import torch.backends.cudnn as cudnn


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, num_layers):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim * 2

        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=embedding_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        x, (_, _) = self.lstm1(x)
        x, (h_0, c_0) = self.lstm2(x) # last time step cell

        return h_0.reshape((batch_size, self.embedding_dim)) # (batch_size, 128)

class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim, n_features, num_layers):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = embedding_dim
        self.n_features = n_features
        self.hidden_dim = embedding_dim * 2

        self.lstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.repeat(self.seq_len,1)
        x = x.reshape((batch_size, self.seq_len, self.input_dim)) # batch_size * 140 * 128
        x, (h_0, c_0) = self.lstm1(x)
        x, (h_0, c_0) = self.lstm2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim)) # 140 * 256

        return self.output_layer(x)


class LstmAutoEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, num_layers):
        super(LstmAutoEncoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim, num_layers).cuda()
        self.decoder = Decoder(seq_len, embedding_dim, n_features,num_layers).cuda()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


# if __name__ == "__main__":
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     traindir = r'/daintlab/data/cfast/only_noised_sensor_dataset/trainset'
#     testdir = r'/daintlab/data/cfast/only_noised_sensor_dataset/testset'

#     tr_ds = Firedata_train(traindir)
#     tr_dl = DataLoader(tr_ds, shuffle=True, batch_size=128)

#     ts_ds = Firedata_test(traindir,testdir)
#     ts_dl = DataLoader(ts_ds, shuffle=False, batch_size=128)

#     x = next(iter(tr_dl))
#     # x.shape = 128,10,2
#     embedding_dim = 128
#     num_layers = 1
#     model = LstmAutoEncoder(10, 2, embedding_dim, num_layers).cuda()
#     print(model)
#     print(model(x.float().cuda()).shape)