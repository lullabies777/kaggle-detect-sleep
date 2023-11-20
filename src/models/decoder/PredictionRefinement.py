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
        mode: str
    ):
        super(PredictionRefinement, self).__init__()

        if dilation * (kernel_size - 1) % 2 != 0:
            raise ValueError("Please re-input dilation, kernel_size!!!")
        else:
            padding = (dilation * (kernel_size - 1)) // 2



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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        return self.prediction_refinement(x)


# if __name__ == '__main__':
#     model = PredictionRefinement([512,128],[128,128],5,[2,2],1,1,[False,False],[False,True],'nearest')
#     x = torch.rand((1, 512, 360))
#     print(model)
#     print(model(x).shape)
