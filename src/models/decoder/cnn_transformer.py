import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn_transformer(nn.Module):
    def __init__(
        self, 
        input_dimension: int,
        num_layers: int,
        nheads: int,
        dropout: int,

    ):
        super(cnn_transformer, self).__init__()
        conv_layers = [nn.Conv1d(input_dimension,input_dimension,(num_layers-i)*2-1,stride=1,padding=0) for i in range(num_layers)]
        self.conv_layers = nn.ModuleList(conv_layers)
        
        deconv_layers = [nn.ConvTranspose1d(input_dimension,input_dimension,(num_layers-i)*2-1,stride=1,padding=0) for i in range(num_layers)]
        self.deconv_layers = nn.ModuleList(deconv_layers)
        
        encoder_layers = [nn.TransformerEncoderLayer(input_dimension, nheads, input_dimension*4, dropout) for i in range(num_layers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        layer_norm_layers = [nn.LayerNorm(input_dimension) for i in range(num_layers)]
        layer_norm_layers2 = [nn.LayerNorm(input_dimension) for i in range(num_layers)]
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        
        for conv, transformer_layer, layer_norm1, layer_norm2, deconv in zip(self.conv_layers,
                                                               self.transformer_encoder,
                                                               self.layer_norm_layers,
                                                               self.layer_norm_layers2,
                                                               self.deconv_layers):
            #LXBXC to BXCXL
            res=x
            x=F.relu(conv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm1(x)
            x=transformer_layer(x)
            x=F.relu(deconv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm2(x)
            x=res+x
        
        return x


# if __name__ == '__main__':
#     model = cnn_transformer(64,4,4,0.1)
#     x = torch.rand((1000,16,64))
#     print(model)
#     print(model(x).shape)
