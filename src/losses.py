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
    
    def forward(self, pred, target, mask=None):
        loss_per_pixel = -(target * torch.log(pred+1e-8) + (1-target) * torch.log(1-pred+1e-8))
        
        if mask != None:
            mask_W, mask_B = mask * target, mask * (1-target)
        else:
            mask_W, mask_B = target, (1-target)

        W, B = torch.sum(mask_W), torch.sum(mask_B)
        loss_W = torch.sum(mask_W * loss_per_pixel) / W
        loss_B = torch.sum(mask_B * loss_per_pixel) / B
            
        return loss_W + loss_B
    
class SketchNoiseMaskLoss(nn.Module):
    def __init__(self, threshold_W):
        super().__init__()
        
        self.threshold_W = threshold_W
        self.sketch_loss = SketchMaskLoss()
    
    def forward(self, pred, target, infodraw):
        mask = infodraw < self.threshold_W # infodraw에서 1이 아닌 부분 
        return self.sketch_loss(pred, target, mask=mask)

class MaskedBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred, target, mask=None):
        loss_per_pixel = -(target * torch.log(pred+1e-8) + (1-target) * torch.log(1-pred+1e-8))
        masked_loss_per_pixel = mask * loss_per_pixel
        loss = torch.sum(masked_loss_per_pixel) / torch.sum(mask)

        return loss
        
        