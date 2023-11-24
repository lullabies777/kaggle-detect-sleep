from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def transformer_encoder_model(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation="relu"
):
    """
    创建一个标准的 Transformer 模型。

    参数:
    - d_model: 模型的特征维度
    - nhead: 多头注意力机制中的头数
    - num_encoder_layers: 编码器中的层次数量
    - dim_feedforward: 前馈网络模型的维度
    - dropout: Dropout比例
    - activation: 激活函数类型
    
    (batch, seq, feature) -- >  (batch, seq, feature)
    返回:
    - 一个 Transformer 模型
    """
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        batch_first=True
    )
    transformer_encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_encoder_layers
    )
    return transformer_encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
class ContextDetection(nn.Module):
    '''
    中间的rnn层
    '''
    def __init__(
        self,
        fe_fc_dimension: int,
        lstm_dimensions: list[int],
        num_layers: list[int],
        n_head: int,
        dropout: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        encoder_type: str,
        activation: str,
        sequence_length,
        chunks
    ):
        super(ContextDetection, self).__init__()
        self.fe_fc_dimension = fe_fc_dimension
        self.num_layers = num_layers
        self.lstm_dimensions = lstm_dimensions
        self.encoder_output_dimension = self.lstm_dimensions[-1] * 2
        self.encoder_type = encoder_type
        if num_layers != len(lstm_dimensions):
            print(f"num layers of lstm is {num_layers}")
            print(f"lstm_dimensions of lstm is {lstm_dimensions}")
            self.num_layers = len(self.lstm_dimensions)

        if encoder_type == "lstm":
            context_detection = []
            for i in range(self.num_layers):
                if i == 0:
                    context_detection.extend([
                        nn.LSTM(
                            input_size=self.fe_fc_dimension,
                            hidden_size=self.lstm_dimensions[i],
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True,
                        )
                    ])

                else:
                    context_detection.extend([
                        nn.LSTM(
                            input_size=self.lstm_dimensions[i - 1] * 2,
                            hidden_size=self.lstm_dimensions[i],
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True
                        )
                    ])
            self.context_detection = nn.Sequential(*context_detection)
            self.encoder_output_dimension = self.lstm_dimensions[-1] * 2
            
        if encoder_type == "transformer":
            # 中间Transformer层
            self.transformer_encoder = transformer_encoder_model(
                d_model=self.fe_fc_dimension,
                nhead=n_head,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            self.positional_encoding = PositionalEncoding(d_model=self.fe_fc_dimension)
            self.encoder_output_dimension = self.fe_fc_dimension
            self.context_detection = self.transformer_encoder
        
        self.inter_upsample = nn.Upsample(
            scale_factor=sequence_length // chunks,
            mode='nearest'
        )
        self.inter_upsample_di = nn.Upsample(
            scale_factor=sequence_length // chunks // 2,
            mode='nearest'
        )
        self.inter_fc = nn.Linear(in_features=self.encoder_output_dimension, out_features=3)
    def forward(self, x: torch.Tensor):
        if self.encoder_type  == "transformer":
            #x = self.positional_encoding(x)
            x = self.context_detection(x)
            #print(f"layer output is {x.shape}")
        elif self.encoder_type  == "lstm":
            for layer in self.context_detection:
                x, _ = layer(x)  #(output, (h_n, c_n))
                #print(f"layer output is {x.shape}")
                
        left = LeftBranch(upsample = self.inter_upsample, fc = self.inter_fc)
        output1 = left(x)

        di = x.permute(0, 2, 1)
        #print(f"before up sample is {di.shape}")
        di = self.inter_upsample_di(di)
        return output1, di

class LeftBranch(nn.Module):
    def __init__(self, upsample, fc):
        super(LeftBranch, self).__init__()
        self.upsample = upsample
        self.fc = fc

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x

# if __name__ == '__main__':
#     model = ContextDetection([64,256],[128,128],[2,2],[True,True],1024,6)
#     x = torch.rand((1,4,64))
#     print(model)
#     print(model(x)[0].shape,model(x)[1].shape)
