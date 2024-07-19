import os
import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    recall_score, 
    confusion_matrix, 
    precision_score,
)



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
        
def inference_logits(model, vtf, img):
    model = model.eval()
    
    with torch.no_grad():
        pred_target = model(vtf, img)  # [B x W x H x 1]

        # Apply sigmoid to get probabilities
        pred_target = torch.sigmoid(pred_target)

    return pred_target

def inference(model, vtf, img):
    model = model.eval()
    
    with torch.no_grad():
        pred_target = model(vtf, img)  # [B x W x H x 1]

        # Apply sigmoid to get probabilities
        pred_target = torch.sigmoid(pred_target)

        # Convert probabilities to binary predictions (threshold at 0.5)
        pred_target = (pred_target > 0.5).float()

    return pred_target

def calculate_noise_metric(preds, targets):
    ####### sketch metric ###### 
    # sketch: 1, Background: 0 #
    ############################
    
    # 확률 값을 0과 1로 분류 (0.5 초과는 1, 나머지는 0)
    preds = preds > 0.5 
    
    # sketch 는 실제 이미지에서 0, BG는 1 이므로 이 둘을 서로 바꿔줘야함.
    sm_preds   = 1 - preds 
    sm_targets = 1 - targets
    
    accuracy = accuracy_score(y_pred=sm_preds, y_true=sm_targets)
    f1score  = f1_score(y_pred=sm_preds, y_true=sm_targets)
    recall   = recall_score(y_pred=sm_preds, y_true=sm_targets)
    conf_mat = confusion_matrix(y_pred=sm_preds, y_true=sm_targets)
    prec     = precision_score(y_pred=sm_preds, y_true=sm_targets)
    sketch_class_acc = np.sum((sm_targets == 1) * sm_preds) / np.sum(sm_targets == 1)
    non_sketch_class_acc = np.sum((sm_targets == 0) * (1-sm_preds)) / np.sum(sm_targets == 0)
    sketch_non_sketch_avrg_acc = (sketch_class_acc + non_sketch_class_acc) / 2
    
    return {
        'accuracy': accuracy,
        'f1score': f1score,
        'recall': recall,
        'confusion_matrix': conf_mat,
        'precision': prec,
        'sketch_class_acc': sketch_class_acc,
        'non_sketch_class_acc': non_sketch_class_acc,
        'sketch_non_sketch_avrg_acc': sketch_non_sketch_avrg_acc,
    }
    

if __name__ == "__main__":
    a = np.array([False, True, True])
    b = np.array([0, 1, 0])
    
    print(a.shape, b.shape)
    
    metric = calculate_noise_metric(preds=a, targets=b)
    print(metric)
    