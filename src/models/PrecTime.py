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

        self.fc1 = nn.Linear(
            256 * (cfg.sequence_length // cfg.chunks // 2), 64
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

    def forward(self, x,labels: Optional[torch.Tensor] = None):
        if x.shape[-1] % self.cfg.chunks != 0:
            print(ValueError("Sequence_Length Should be Divided by Num_Chunks"))
        x = x.reshape(
            -1,
            len(self.cfg.features),
            x.shape[-1] // self.cfg.chunks
        )
        print("The shape put into feature extraction:", x.shape)
        features_combined = self.feature_extractor(x)
        features_combined_flat = features_combined.view(1, self.cfg.chunks, -1)
        print("The shape after the flatten of concat output:",
              features_combined_flat.shape)
        
        features_combined_flat = self.fc1(features_combined_flat)
        print("The shape after using fc to reduce dimension:",
              features_combined_flat.shape)
        output1, di = self.encoder(features_combined_flat)
        ui = features_combined.reshape(1, features_combined.shape[1], -1)
        print("The shape after Reshaping Ui:", ui.shape)
        combine_ui_di = torch.cat([ui, di], dim=1)
        print("The shape after combining Ui and Di:", combine_ui_di.shape)

        final_output = self.decoder(combine_ui_di)
        final_output = self.fc_final(final_output.permute(0, 2, 1))

        return final_output

        

        