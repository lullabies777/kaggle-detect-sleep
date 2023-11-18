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
        input_size: list[int] ,
        hidden_size: list[int],
        num_layers: list[int],
        bidirectional: list[bool],
        sequence_length,
        chunks
    ):
        super(ContextDetection, self).__init__()
        context_detection = []
        for index in range(len(input_size)):
            tmp_input_size = input_size[index]
            tmp_hidden_size = hidden_size[index]
            tmp_num_layers = num_layers[index]
            tmp_bidirectional = bidirectional[index]
            tmp_layer = nn.LSTM(
                input_size = tmp_input_size,
                hidden_size = tmp_hidden_size,
                num_layers = tmp_num_layers,
                bidirectional = tmp_bidirectional,
                batch_first=True)
            context_detection.append(tmp_layer)
        self.context_detection = nn.Sequential(*context_detection)
        self.inter_upsample = nn.Upsample(
            scale_factor=sequence_length // chunks,
            mode='nearest'
        )
        self.inter_upsample_di = nn.Upsample(
            scale_factor=sequence_length // chunks // 2,
            mode='nearest'
        )
        self.inter_fc = nn.Linear(in_features=context_detection[-1].hidden_size * 2, out_features=3)
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
