import torch
import torch.nn as nn



class FocalLoss(nn.Module):
    # https://woochan-autobiography.tistory.com/929
    
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(pred, targets)
        
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        return torch.mean(F_loss)

class SketchMaskLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred, target):
        p = torch.sigmoid(pred)

        loss_per_pixel = -(target * torch.log(p) + (1-target) * torch.log(1-p))
        loss = (1/torch.sum(target)) * torch.sum(target * loss_per_pixel) + (1/torch.sum(1-target)) * torch.sum((1-target) * loss_per_pixel)
        return loss