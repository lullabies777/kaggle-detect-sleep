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

import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, maxpool=False):
        super(ResidualBlock, self).__init__()
        self.maxpool = maxpool
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Adjusting downsample to handle different in_channels and out_channels
        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm1d(out_channels)
            )

        if self.maxpool:
            self.maxpool_layer = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        if self.maxpool:
            out = self.maxpool_layer(out)

        return out



class PrecFeatureExtractor(nn.Module):
    '''
    CNN1d For feature extraction

    return U_i
    '''
    def __init__(
        self,
        input_channels: int,
        left_hidden_channels: list[int],
        right_hidden_channels: list[int],
        left_fe_kernel_size: int,
        right_fe_kernel_size: int,
        left_fe_padding: int,
        right_fe_padding: int,
        left_fe_stride: int,
        right_fe_stride: int,
        left_fe_dilation: int,
        right_fe_dilation: int,
        cnn_name:str
    ):
        super(PrecFeatureExtractor, self).__init__()
        self.input_channels = input_channels

        self.left_hidden_channels = left_hidden_channels
        self.right_hidden_channels = right_hidden_channels

        self.left_fe_kernel_size = left_fe_kernel_size
        self.right_fe_kernel_size = right_fe_kernel_size

        self.left_fe_padding = left_fe_padding
        self.right_fe_padding = right_fe_padding


        self.left_fe_stride = left_fe_stride
        self.right_fe_stride = right_fe_stride

        self.left_fe_dilation = left_fe_dilation
        self.right_fe_dilation = right_fe_dilation

        self.fe1_layers = len(self.left_hidden_channels)
        self.fe2_layers = len(self.right_hidden_channels)

        #check padding
        if self.left_fe_dilation * (self.left_fe_kernel_size - 1) % 2 != 0:
            raise ValueError("Please re-input left dilation, kernel_size!!!")
        else:
            self.left_fe_padding = (self.left_fe_dilation * (self.left_fe_kernel_size - 1)) // 2
        
        if self.right_fe_dilation * (self.right_fe_kernel_size - 1) % 2 != 0:
            raise ValueError("Please re-input right dilation, kernel_size!!!")
        else:
            self.right_fe_padding = (self.right_fe_dilation * (self.right_fe_kernel_size - 1)) // 2
        name = 'r_cnn1d'
        
        # 左侧特征提取分支
        feature_extraction1_layer = []
        feature_extraction1_layer.extend([
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.left_hidden_channels[0],
                kernel_size=self.left_fe_kernel_size,
                padding=self.left_fe_padding,
                stride=self.left_fe_stride,
                dilation=self.left_fe_dilation,
                maxpool=True,
                dropout=True
            )]
        )
        if name == 'cnn1d':
            for i in range(self.fe1_layers - 1): 
                feature_extraction1_layer.extend([
                    conv1d_block(
                        in_channels=self.left_hidden_channels[i],
                        out_channels=self.left_hidden_channels[i + 1],
                        kernel_size=self.left_fe_kernel_size,
                        padding=self.left_fe_padding,
                        stride=self.left_fe_stride,
                        dilation=self.left_fe_dilation
                    )
                ])
        
        if name == 'r_cnn1d':
            for i in range(1, len(left_hidden_channels)):
                feature_extraction1_layer.append(ResidualBlock(
                    self.left_hidden_channels[i - 1], self.left_hidden_channels[i], self.left_fe_kernel_size, self.left_fe_stride, self.left_fe_padding, self.left_fe_dilation))
        
        self.feature_extraction_left = nn.Sequential(
            *feature_extraction1_layer
        )

        # 右侧特征提取分支
        feature_extraction2_layer = []
        feature_extraction2_layer.extend([
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.right_hidden_channels[0],
                kernel_size=self.right_fe_kernel_size,
                padding=self.right_fe_padding,
                stride=self.right_fe_stride,
                dilation=self.right_fe_dilation,
                maxpool=True,
                dropout=True
            )]
        )
        if name == 'cnn1d':
            for i in range(self.fe2_layers - 1):
                feature_extraction2_layer.extend([
                    conv1d_block(
                        in_channels=self.right_hidden_channels[i],
                        out_channels=self.right_hidden_channels[i + 1],
                        kernel_size=self.right_fe_kernel_size,
                        padding=self.right_fe_padding,
                        stride=self.right_fe_stride,
                        dilation=self.right_fe_dilation
                    )
                ])

        if name == 'r_cnn1d':
            for i in range(1, len(right_hidden_channels)):
                feature_extraction2_layer.append(ResidualBlock(
                    self.right_hidden_channels[i - 1], self.right_hidden_channels[i], self.right_fe_kernel_size, self.right_fe_stride, self.right_fe_padding, self.right_fe_dilation))
                
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
