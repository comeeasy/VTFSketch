import os
import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix



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
    
    return {
        'accuracy': accuracy,
        'f1score': f1score,
        'recall': recall,
        'confusion_matrix': conf_mat,
    }
    
def point_interpolate_from_gray_image(x: float, y: float, img):
    # img(gray scale) -> [H, W]
    
    A_pos = int(x), int(y)
    B_pos = int(x)+1, int(y)
    C_pos = int(x), int(y)+1
    D_pos = int(x)+1, int(y)+1
    A, B, C, D = img[A_pos[1], A_pos[0]], img[B_pos[1], B_pos[0]], img[C_pos[1], C_pos[0]], img[D_pos[1], D_pos[0]]
    alpha = x - A_pos[0]
    beta = x - C_pos[0]
    gamma = y - A_pos[1]
    AB = alpha * B + (1-alpha) * A
    CD = beta * D + (1-beta) * C
    P = gamma * CD + (1-gamma) * AB
    
    return P

def cosine_similarity(x, y):
    x, y = np.float64(x), np.float64(y)
    return np.dot(x, y) / max((np.linalg.norm(x) * np.linalg.norm(y)), np.finfo(float).eps)    

def get_curvature(flowpath):
    curvature = []
    for i in range(1, 20):
        curvature_i = flowpath[i+1] - flowpath[i-1]
        curvature.append(curvature_i)
    return np.array(curvature, dtype=np.float64)

def calc_sim_geom(curvature0, curvature1):
    # uses cosine similarity
    sim_geoms = []
    for cvt0, cvt1 in zip(curvature0, curvature1):
        sim_geom = cosine_similarity(cvt0, cvt1)
        sim_geoms.append(sim_geom)
    return np.mean(sim_geoms)

def calc_sim_color(vtf_infodraw, vtf_target):
    # use cosine similarity
    # sim_color = cosine_similarity(vtf_infodraw, vtf_target)
    
    sim_color = np.mean(1-vtf_target)
    return sim_color

def calc_sim_color_ssd(vtf1, vtf2):
    return np.mean((vtf1 - vtf2)*(vtf1 - vtf2))

def calc_sim_color_sad(vtf1, vtf2):
    return np.mean(np.abs(vtf1 - vtf2))

def get_vtf_target(flowpath, target):
    intensities = []
    for x, y in flowpath:
        intensity = point_interpolate_from_gray_image(x, y, target)
        intensities.append(intensity)
    return np.array(intensities)
    
    