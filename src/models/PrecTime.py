from typing import Optional

import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.losses import get_loss

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
        self.fc1 = nn.Linear(self.cfg.feature_extractor.hidden_channels * 2 * (self.cfg.sequence_length // self.cfg.chunks // 2), 64
        )

        self.inter_upsample = nn.Upsample(
            scale_factor=cfg.sequence_length // cfg.chunks,
            mode='nearest'
        )
        self.inter_fc = nn.Linear(in_features=256, out_features=3)

        self.inter_upsample_di = nn.Upsample(
            scale_factor=cfg.sequence_length // cfg.chunks // 2,
            mode='nearest'
        )

        self.fc_final = nn.Linear(128, 3)
        self.loss_fn = get_loss(cfg)

    def forward(
        self, 
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False):
        origin_x = x
        #print(f"input shape is: {origin_x.shape}")
        if x.shape[-1] % self.cfg.chunks != 0:
            print(ValueError(f"Sequence_Length Should be Divided by Num_Chunks, Sequence_Length is {x.shape[-1]}"))

        x = x.transpose(0, 1).reshape(
            x.shape[1], -1, x.shape[2] // self.cfg.chunks
        ).transpose(0, 1)

        #print("The shape put into feature extraction:", x.shape)
        features_combined = self.feature_extractor(x)
        #print("The shape after the flatten of concat output:", features_combined.shape)
        features_combined_flat = features_combined.view(origin_x.shape[0], self.cfg.chunks, -1) # (batzh, chunks, flaten_features)
        #print("The shape after the flatten of concat output:",features_combined_flat.shape)
        
        features_combined_flat = self.fc1(features_combined_flat)
        #print("The shape after using fc to reduce dimension:",features_combined_flat.shape)
        output1, di = self.encoder(features_combined_flat)
        #print(f"The first output is {output1.shape}")

        ui = features_combined.transpose(0, 1).reshape(
            features_combined.shape[1], origin_x.shape[0], -1
        ).transpose(0, 1)
        #print("The shape after Reshaping Ui:", ui.shape)
        combine_ui_di = torch.cat([ui, di], dim=1)
        #print("The shape after combining Ui and Di:", combine_ui_di.shape)

        final_output = self.decoder(combine_ui_di) # (batch, chanels, timestamps)
        #print("The shape after prediction refinement:", final_output.shape)
        final_output = self.fc_final(final_output.permute(0, 2, 1)) # (batch, timestamps, n_classes)
        #print("The final shape after fc:", final_output.shape)

        logits = final_output
        # reduce overlap_interval 
        #print(f"logist shape is {logits.shape}") 
        logits = logits[:, (self.cfg.overlap_interval // self.cfg.downsample_rate) : logits.shape[1] - (self.cfg.overlap_interval // self.cfg.downsample_rate), :]
        #print(f"labels shape is {labels.shape}")
        output = {"logits": logits}
        if labels is not None:
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

        

        