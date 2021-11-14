import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from attention import Attention
import numpy as np

class LSTM(nn.Module):
    def __init__(self, embedding_matrix, batch_size, seq_length, hidden_size, layers):
        super(LSTM, self).__init__()
        num_words = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        self.hidden_size = hidden_size
        self.layers = layers
        self.batch_size = batch_size
        self.embedding = nn.Embedding(num_embeddings = num_words, embedding_dim=embed_dim)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embed_dim, self.hidden_size, self.layers, batch_first=True, bidirectional = True)
        self.atten1 = Attention(self.hidden_size*2, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size, self.layers, batch_first = True, bidirectional = True)
        self.atten2 = Attention(self.hidden_size*2, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        # next_layer = seq_length*self.hidden_size*2
        ## Double the linear size if bidirectional (128*2)
        self.pre_out = nn.Linear(self.hidden_size*2*2, self.hidden_size*2*2)
        self.RELU = nn.ReLU()
        self.out = nn.Linear(self.hidden_size*2*2, 1)
        # self.out = nn.Linear(512, 1)
 
    def forward(self, x, hidden): #, hidden
        # print('Size of X before embedding', x.size())
        batch_size = x.size(0)
        
        x_len = torch.Tensor(np.array([158 for i in range(0, 64)]))
        # print(x_len.size())
        # h0 = torch.nn.init.xavier_uniform(torch.Tensor(self.layers*2, x.size(0), self.hidden_size).cuda())
        # c0 = torch.nn.init.xavier_uniform(torch.Tensor(self.layers*2, x.size(0), self.hidden_size).cuda())
        x = self.embedding(x)
        x = self.dropout(x)
        
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out, hidden = self.lstm(x, hidden)
        
        x, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        x, _ = self.atten1(x, lengths)

        out, _ = self.lstm2(out)
        
        y, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        y, _ = self.atten2(y, lengths)
        
        # avg_pool = torch.mean(out, 1)
        # max_pool, _ = torch.max(out, 1)

        out = torch.cat([x, y], 1)

        out = self.dropout(out)
        
        # print('Shape of out', out.size())

        out = self.RELU(self.pre_out(out))

        # print('Shape of out again', out.size())
        
        out = self.out(out)

        # print('Shape of out again and again', out.size())
        return out, hidden
        
    def init_hidden(self, batch_size, device, model):
        weight = next(self.parameters()).data
        a = weight.new(self.layers*2, batch_size, self.hidden_size)#.zero_()#uniform_(-1, 1) * math.sqrt(1./batch_size)
        b = torch.nn.init.xavier_uniform(a)
        hidden = (b.to(device), b.to(device))
        return hidden