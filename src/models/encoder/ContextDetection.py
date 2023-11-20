from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextDetection(nn.Module):
    '''
    中间的rnn层
    '''
    def __init__(
        self,
        fe_fc_dimension: int,
        lstm_dimensions: list[int],
        num_layers: list[int],
        bidirectional: list[bool],
        sequence_length,
        chunks
    ):
        super(ContextDetection, self).__init__()
        self.fe_fc_dimension = fe_fc_dimension
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.lstm_dimensions = lstm_dimensions
        self.encoder_output_dimension = self.lstm_dimensions[-1] * 2
        
        if num_layers != len(lstm_dimensions):
            print(f"num layers of lstm is {num_layers}")
            print(f"lstm_dimensions of lstm is {lstm_dimensions}")
            self.num_layers = len(self.lstm_dimensions)

        

        context_detection = []
        for i in range(self.num_layers):
            if i == 0:
                context_detection.extend([
                    nn.LSTM(
                        input_size=self.fe_fc_dimension,
                        hidden_size=self.lstm_dimensions[i],
                        num_layers=1,
                        bidirectional=True
                    )
                ])

            else:
                context_detection.extend([
                    nn.LSTM(
                        input_size=self.lstm_dimensions[i - 1] * 2,
                        hidden_size=self.lstm_dimensions[i],
                        num_layers=1,
                        bidirectional=True
                    )
                ])
        self.context_detection = nn.Sequential(*context_detection)
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
        for layer in self.context_detection:
            x, _ = layer(x)  #(output, (h_n, c_n))
            #print(f"layer output is {x.shape}")
        # 有softmax的
        left = LeftBranch(upsample = self.inter_upsample, fc = self.inter_fc)
        output1 = left(x)

        di = x.permute(0, 2, 1)
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
