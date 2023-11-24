from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ResidualLSTM(nn.Module):

    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM=nn.LSTM(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1=nn.Linear(d_model*2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)


    def forward(self, x):
        res=x
        x, _ = self.LSTM(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=res+x
        return x
    
class ResidualGRU(nn.Module):

    def __init__(self, d_model):
        super(ResidualGRU, self).__init__()
        self.GRU=nn.GRU(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear=nn.Linear(d_model*2, d_model)


    def forward(self, x):
        res=x
        x, _ = self.GRU(x)
        x=self.linear(x)
        x=res+x
        return x
    
class Residual_lstm(nn.Module):
    '''
    中间的rnn层
    '''
    def __init__(
        self,
        input_dimension: int,
        num_layers: int,
        dropout: int,
        names : str
    ):
        super(Residual_lstm, self).__init__()
                    
        if names == "lstm":
            self.pos_encoder = nn.ModuleList([ResidualLSTM(input_dimension) for i in range(num_layers)])
        elif names == "gru":
            self.pos_encoder = nn.ModuleList([ResidualGRU(input_dimension) for i in range(num_layers)])
        self.pos_encoder_dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(input_dimension)
    def forward(self, x: torch.Tensor):
        for lstm in self.pos_encoder:
            lstm.LSTM.flatten_parameters()
            x=lstm(x)
        x = self.pos_encoder_dropout(x)
        x = self.layer_normal(x)
        
        return x 



# if __name__ == '__main__':
#     model = Residual_lstm(4,4,0.1,'lstm')
#     x = torch.rand((16,64,4))
#     print(model)
#     print(model(x)[0].shape,model(x)[1].shape)
