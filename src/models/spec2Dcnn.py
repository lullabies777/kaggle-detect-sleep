from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.losses import get_loss

class Spec2DCNN(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder: nn.Module,
        cfg,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.decoder = decoder
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = get_loss(cfg)
        self.cfg = cfg

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        #print(f'input shape is {x.shape}') # [16, 4, 7296]
        assert x.shape[-1] == (self.cfg.duration + 2 * self.cfg.overlap_interval), f"x shape: {x.shape}, duration: {self.cfg.duration}, overlap_interval: {self.cfg.overlap_interval}"

        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
        logits = self.decoder(x)  # (batch_size, n_timesteps, n_classes)

        # reduce overlap_interval 
        # [:768/2 : shape - (768/2), :] = 2880
        # print(f"logist shape is {logits.shape}") # [16, 3648, 3]
        logits = logits[:, (self.cfg.overlap_interval // self.cfg.downsample_rate) : logits.shape[1] - (self.cfg.overlap_interval // self.cfg.downsample_rate), :]
        # print(f"labels shape is {labels.shape}")# [16, 2880, 3]
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
