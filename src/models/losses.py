
import torch
import torch.nn as nn
import torch.nn.functional as F 
from omegaconf import DictConfig

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss
    
def get_loss(
    cfg: DictConfig
):
    if cfg.loss.name == "bce":
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')
    elif cfg.loss.name == "mse":
        criterion = nn.MSELoss(reduction = 'none')
    elif cfg.loss.name == "focal":
        criterion = FocalLoss(alpha = cfg.loss.alpha, gamma = cfg.loss.gamma)
    elif cfg.loss.name == "kl":
        criterion = nn.KLDivLoss(reduction = 'none')
    else:
        raise ValueError(f"Invalid loss name: {cfg.loss.name}")
    
    return criterion