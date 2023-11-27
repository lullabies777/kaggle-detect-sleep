import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.decoder.cnn_transformer import cnn_transformer
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, maxpool=False,dropout=False):
        super(ResidualBlock, self).__init__()
        self.maxpool = maxpool
        self.dropout = dropout
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
        if self.dropout:
            self.dropout_layer = nn.Dropout(p=0.5)
            
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
        if self.dropout:
            out = self.dropout_layer(out)
        return out
            
class PredictionRefinement(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        dilation: int,
        scale_factor: int,
        mode: str,
        cnn_name: str
    ):
        super(PredictionRefinement, self).__init__()

        if dilation * (kernel_size - 1) % 2 != 0:
            raise ValueError("Please re-input dilation, kernel_size!!!")
        else:
            padding = (dilation * (kernel_size - 1)) // 2
            
        self.cnn_name = cnn_name
        
        if cnn_name == 'conv1d':
            self.prediction_refinement = nn.Sequential(
                conv1d_block(
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                    maxpool=False,
                    dropout=False
                ),
                nn.Upsample(scale_factor=scale_factor, mode=mode),
                conv1d_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                    maxpool=False,
                    dropout=True
                ),
                nn.Dropout(p=0.5)
            )
        if cnn_name == 'r_conv1d':
            self.prediction_refinement = nn.Sequential(
                ResidualBlock(
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                    maxpool=False,
                    dropout=False
                ),
                nn.Upsample(scale_factor=scale_factor, mode=mode),
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                    maxpool=False,
                    dropout=True
                ),
                nn.Dropout(p=0.5)
            )
        
        elif cnn_name == 'cnn_transformer':
            self.prediction_refinement = cnn_transformer(
                input_dimension= in_channels,
                num_layers= 2,
                nheads= 8,
                dropout= 0.5,
            )
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        if self.cnn_name == 'cnn_transformer':
            # (batch, features, length) -> (length,batch, features)
            x = x.permute(2,0,1)
            x = self.prediction_refinement(x)
            x = x.permute(1,2,0)
            x = self.upsample(x)
        else:
            x = self.prediction_refinement(x)
        return x


# if __name__ == '__main__':
#     model = PredictionRefinement([512,128],[128,128],5,[2,2],1,1,[False,False],[False,True],'nearest')
#     x = torch.rand((1, 512, 360))
#     print(model)
#     print(model(x).shape)
