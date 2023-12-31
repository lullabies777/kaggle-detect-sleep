from typing import Optional

import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.losses import get_loss
import torch.nn.functional as F
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
    
class PrecTime(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        cfg,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.decoder = decoder
        self.cfg = cfg
        self.fc_after_fe = nn.Linear(
            (self.cfg.feature_extractor.left_hidden_channels[-1] + self.cfg.feature_extractor.right_hidden_channels[-1]) *
            (self.cfg.sequence_length // self.cfg.chunks // 2), self.cfg.encoder.fe_fc_dimension
        )

        self.fc_final = nn.Linear(self.cfg.decoder.out_channels, self.cfg.num_classes)
        if self.cfg.decoder.cnn_name == 'cnn_transformer':
            self.fc_final = nn.Linear(self.cfg.decoder.in_channels, self.cfg.num_classes)
        else:
            self.fc_final = nn.Linear(self.cfg.decoder.out_channels, self.cfg.num_classes)
        self.loss_fn = get_loss(cfg)
        
        # input_dimension = len(cfg.features)
        # self.pos_encoder = nn.ModuleList([ResidualLSTM(input_dimension) for i in range(2)])
        # self.pos_encoder_dropout = nn.Dropout(0.5)
        # self.layer_normal = nn.LayerNorm(input_dimension)
        
    def forward(
        self, 
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False):
        origin_x = x
        #print(f"input shape is: {origin_x.shape}")
        #input : (batch, features, length)
        if x.shape[-1] % self.cfg.chunks != 0:
            raise ValueError(f"Sequence_Length Should be Divided by Num_Chunks, Sequence_Length is {x.shape[-1]}")
            
        if (x.shape[-1] / self.cfg.chunks) % 2 != 0:
            raise ValueError(f"Sequence_Length Divided by Num_Chunks should be 2 times X, "
                     f"Sequence_Length Divided by Num_Chunks is {x.shape[-1]} / {self.cfg.chunks} = {x.shape[-1] / self.cfg.chunks}")
    
        x = x.transpose(0, 1).reshape(
            x.shape[1], -1, x.shape[2] // self.cfg.chunks
        ).transpose(0, 1) # (batzh*chunks, features, seq_length//chunks)
        
        # x=x.permute(2,0,1)# ( seq_length//chunks,batzh*chunks, features)
        # for lstm in self.pos_encoder:
        #     lstm.LSTM.flatten_parameters()
        #     x=lstm(x)
        # x = self.pos_encoder_dropout(x)
        # x = self.layer_normal(x)
        # x = x.permute(1,2,0)
        #print("The shape put into feature extraction:", x.shape)
        features_combined = self.feature_extractor(x) # (batzh*chunks, features, length)
        #print("The shape of concat output:", features_combined.shape)
        
        #每个chunks的features
        features_combined_flat = features_combined.view(origin_x.shape[0], self.cfg.chunks,-1) # (batzh, chunks, features)
        #print("The shape after the flatten of concat output:",features_combined_flat.shape)
        
        features_combined_flat = self.fc_after_fe(features_combined_flat) # (batzh, chunks, features)
        #print("The shape after using fc to reduce dimension:",features_combined_flat.shape)
        
        #捕捉chunks之间的关系
        output1, di = self.encoder(features_combined_flat) # (batch, features, length)
        #print(f"The first output is {output1.shape}")
        #print(f"di is {di.shape}")
        ui = features_combined.transpose(0, 1).reshape(
            features_combined.shape[1], origin_x.shape[0], -1
        ).transpose(0, 1) # (batch, features, length)
        #print("The shape after Reshaping Ui:", ui.shape) 
        combine_ui_di = torch.cat([ui, di], dim=1) #(batch, features, length)
        #print("The shape after combining Ui and Di:", combine_ui_di.shape)

        final_output = self.decoder(combine_ui_di) # (batch, features, length)
        #print("The shape after prediction refinement:", final_output.shape)
        final_output = self.fc_final(final_output.permute(0, 2, 1)) # (batch, length, n_classes)
        #print("The final shape after fc:", final_output.shape)
        
        logits = final_output
        logits1 = output1
        # reduce overlap_interval 
        #print(f"logist shape is {logits.shape}") 
        logits = logits[:, (self.cfg.overlap_interval) : logits.shape[1] - (self.cfg.overlap_interval), :]
        logits1 = logits1[:, (self.cfg.overlap_interval) : logits1.shape[1] - (self.cfg.overlap_interval), :]
        #print(f"labels shape is {labels.shape}")
        output = {"logits": logits*0.9 + logits1*0.1}
        #{"logits": logits*0.9 + logits1*0.1}
        if labels is not None:
            #labels = labels[:,:,1:]
            assert logits.shape == labels.shape, f"logits shape: {logits.shape}, labels shape: {labels.shape}"
            
            weight = labels.clone() * self.cfg.loss.weight 
            weight += 1.0
            
            if self.cfg.loss.name == "kl":
                # labels = labels / labels.sum(dim = -1, keepdim = True)
                labels = labels.softmax(dim = -1)
                # logits = logits.sigmoid()
                # logits = logits / logits.sum(dim = -1, keepdim = True)
                logits = logits.softmax(dim = -1)
                
                loss = self.loss_fn(logits.log(), labels)
                    
            else:
                loss = self.loss_fn(logits, labels)*0.9 + self.loss_fn(logits1, labels) * 0.1

            loss = loss * weight 
            
            output["loss"] = loss.mean()

        return output

        

        