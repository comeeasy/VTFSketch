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
        # self.w_W = nn.Parameter(torch.tensor(0.5))  # Initialize w_W with a learnable parameter
        self.w_W = 1
        self.w_B = 1
    
    def forward(self, pred, target, mask=None):
        p = torch.sigmoid(pred)

        loss_per_pixel = -(target * torch.log(p+1e-10) + (1-target) * torch.log(1-p+1e-10))
        
        if mask:
            mask_W, mask_B = mask * target, mask * (1-target)
        else:
            mask_W, mask_B = target, (1-target)

        W, B = torch.sum(mask_W), torch.sum(mask_B)
        loss_W = self.w_W * (torch.sum(mask_W * loss_per_pixel) / W)
        loss_B = self.w_B * (torch.sum(mask_B * loss_per_pixel) / B)
            
        print(f"loss_W: {loss_W.item()},  loss_B: {loss_B.item()}")    
            
        return loss_W + loss_B