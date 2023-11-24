from typing import Optional

import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.losses import get_loss

class Combined(nn.Module):
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
        self.loss_fn = get_loss(cfg)
        
        self.embedding = nn.Linear(len(cfg.features), cfg.encoder.input_dimension)
        self.pred = nn.Linear(cfg.encoder.input_dimension, cfg.num_classes)
    def forward(
        self, 
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False):
        origin_x = x
        
        #input : (batch, features, length)
        #print(f"input shape is {x.shape}")
        x = x.permute(0, 2, 1)
        
        x = self.embedding(x)
        #print(f"after embedding is {x.shape}")
        #(batch, length,features) -> (length, batch ,features)
        x = x.permute(1, 0, 2)
        
        #print(f"input to encoder is {x.shape}")
        x = self.encoder(x)
        
        #print(f"input to decoder is {x.shape}")
        final_output = self.decoder(x)
        x = x.permute(1, 0, 2)
        
        #print(f"before pred is {x.shape}")
        final_output = self.pred(x)
        
        logits = final_output

        # reduce overlap_interval 
        #print(f"logist shape is {logits.shape}") 
        logits = logits[:, (self.cfg.overlap_interval) : logits.shape[1] - (self.cfg.overlap_interval), :]

        #print(f"labels shape is {labels.shape}")
        output = {"logits": logits}
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
                loss = self.loss_fn(logits, labels)

            loss = loss * weight 
            
            output["loss"] = loss.mean()

        return output

        

        