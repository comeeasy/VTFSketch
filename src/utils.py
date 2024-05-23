import os

import torch



class BestModelSelector:
    
    def __init__(self, model, args, metric="precision", mode="max"):
        assert mode in ["max", "min"]
        
        self.model = model
        self.args = args
        self.metric = metric
        print(f"{id(model)=}, {id(self.model)=}")
        
        self.mode = mode
        self.prev_best = -1e4 if mode == "max" else 1e4
    
    def update(self, v_metric, epoch, save=True):
        save_path = f"best_model_{self.metric}_{str(v_metric).replace('.', '_')}.pth"
        
        if self.mode == "max":
            if v_metric > self.prev_best:
                self.prev_best = v_metric
                save_path = os.path.join(self.args.checkpoint_dir, save_path)
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved with {self.metric}: {v_metric:.4f}")
        elif self.mode == "min":
            if v_metric < self.prev_best:
                self.prev_best = v_metric
                save_path = os.path.join(self.args.checkpoint_dir, save_path)
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved with {self.metric}: {v_metric:.4f}")
        
def inference_logits(model, vtf):
    model = model.eval()
    
    with torch.no_grad():
        pred_target = model(vtf)  # [B x W x H x 1]

        # Apply sigmoid to get probabilities
        pred_target = torch.sigmoid(pred_target)

    return pred_target

def inference(model, vtf):
    model = model.eval()
    
    with torch.no_grad():
        pred_target = model(vtf)  # [B x W x H x 1]

        # Apply sigmoid to get probabilities
        pred_target = torch.sigmoid(pred_target)

        # Convert probabilities to binary predictions (threshold at 0.5)
        pred_target = (pred_target > 0.5).float()

    return pred_target