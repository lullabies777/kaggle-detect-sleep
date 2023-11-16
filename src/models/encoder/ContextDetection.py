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
        bidirectional: list[bool]
    ):
        super(RNN, self).__init__()
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

    def forward(self, x: torch.Tensor):
        for layer in self.context_detection:
            x, _ = layer(x)  #(output, (h_n, c_n))
        return x

# if __name__ == '__main__':
#     model = RNN([64,256],[128,128],[2,2],[True,True])
#     x = torch.rand((1,4,64))
#     print(model)
#     print(model(x).shape)
