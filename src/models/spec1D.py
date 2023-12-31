from typing import Optional

import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.losses import get_loss


class Spec1D(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        cfg,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.channels_fc = nn.Linear(feature_extractor.out_chans * feature_extractor.height, feature_extractor.height)
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

        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)
        
        # pool over n_channels dimension
        # x = x.transpose(1, 3)  # (batch_size, n_timesteps, height, n_channels)
        # x = self.channels_fc(x)  # (batch_size, n_timesteps, height, 1)
        # x = x.squeeze(-1).transpose(1, 2)  # (batch_size, height, n_timesteps)
        
        x = x.transpose(1, 3).transpose(2, 3).flatten(2)  # (batch_size, n_timesteps, n_channels * height )

        x = self.channels_fc(x)  # (batch_size, n_timesteps, height)

        logits = self.decoder(x.transpose(-1,-2))  # (batch_size, n_classes, n_timesteps)
        
        # reduce overlap_interval 
        logits = logits[:, (self.cfg.overlap_interval // self.cfg.downsample_rate) : logits.shape[1] - (self.cfg.overlap_interval // self.cfg.downsample_rate), :]

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
