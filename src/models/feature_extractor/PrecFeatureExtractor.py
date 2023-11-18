from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1d_block(
    in_channels,
    out_channels,
    kernel_size=5,
    stride=1,
    padding=2,
    dilation=1,
    maxpool=False,
    dropout=False
):
    layers = [nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )]
    if maxpool:
        layers.append(nn.MaxPool1d(kernel_size=2))
    if dropout:
        layers.append(nn.Dropout(p=0.5))
    return nn.Sequential(*layers)

class PrecFeatureExtractor(nn.Module):
    '''
    CNN1d For feature extraction

    return U_i
    '''
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        padding,
        stride,
        dilation,
        fe1_layers=4,
        fe2_layers=4
    ):
        super(PrecFeatureExtractor, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.fe1_layers = fe1_layers
        self.fe2_layers = fe2_layers
        # 左侧特征提取分支
        feature_extraction1_layer = []
        feature_extraction1_layer.extend([
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation
            ),
            conv1d_block(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                maxpool=True,
                dropout=True
            )
        ])
        for i in range(self.fe1_layers):
            feature_extraction1_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation
                )
            ])
        self.feature_extraction_left = nn.Sequential(
            *feature_extraction1_layer
        )

        # 右侧特征提取分支
        self.padding = 8
        feature_extraction2_layer = []
        feature_extraction2_layer.extend([
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=4
            ),
            conv1d_block(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=4,
                maxpool=True,
                dropout=True
            )
        ])
        for i in range(self.fe2_layers):
            feature_extraction2_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=4
                )
            ])
        self.feature_extraction_right = nn.Sequential(
            *feature_extraction2_layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features1 = self.feature_extraction_left(x)
        #print("The output shape from left feature extraction:", features1.shape)
        features2 = self.feature_extraction_right(x)
        #print("The output shape from right feature extraction:", features2.shape)
        features_combined = torch.cat((features1, features2), dim=1)
        return features_combined

# if __name__ == '__main__':
#     model = PrecFeatureExtractor(6,128,5,2,1,1)
#     x = torch.rand((4,6,180,))
#     print(model)
#     print(model(x))
