import os
import torch

import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
import argparse
from datetime import datetime

from src.dataloaders import get_data_loaders
from src.models import FPathPredictor, UNetFPathPredictor
from src.losses import FocalLoss, SketchMaskLoss
from src.utils import inference, inference_logits, BestModelSelector

torch.backends.cudnn.enabled = False

import wandb




def train(model, optimizer, objective, train_loader, accelerator):
    model.train()
    
    cum_loss = 0
    for vtf, img, target in tqdm(train_loader):
        vtf = vtf.to(accelerator.device) 
        img = img.to(accelerator.device)
        target = target.to(accelerator.device)
        
        pred_target = model(vtf, img) 
        
        # mask = vtf[:, 10, :, :] != 1.0 # infodraw
        mask = None
        loss = objective(pred_target, target, mask)
        cum_loss += loss.detach().item() / len(train_loader)
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
    
    wandb.log({"train_loss": cum_loss})
    print(f"train loss: {cum_loss}")

def val(model, objective, val_loader, epoch, accelerator, model_selector: BestModelSelector):
    model.eval()
    
    all_preds = []
    all_targets = []
    cum_loss = 0
    with torch.no_grad():
        for vtf, img, target in tqdm(val_loader):
            vtf = vtf.to(accelerator.device) 
            img = img.to(accelerator.device)
            target = target.to(accelerator.device)
        
            pred_target = inference_logits(model, vtf, img)
        
            # mask = vtf[:, 10, :, :] != 1.0 # infodraw
            mask = None
            loss = objective(pred_target, target, mask)
            cum_loss += loss.detach().item() / len(val_loader)
            
            pre_target_one_zero = inference(model, vtf, img)
            
            images = wandb.Image(pre_target_one_zero, caption="Top: Output, Bottom: Input")
            wandb.log({"val_results": images})
            
            all_preds.append(pre_target_one_zero.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    wandb.log({"val_loss": cum_loss})
    print(f"[EPOCH {epoch}] val loss: {cum_loss}")

    model_selector.update(v_metric=cum_loss, epoch=epoch, save=True)

def test(model, test_loader, accelerator):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for vtf, img, target in tqdm(test_loader):
            vtf = vtf.to(accelerator.device) 
            img = img.to(accelerator.device)
            target = target.to(accelerator.device)
            
            pred_target = inference(model, vtf, img)

            all_preds.append(pred_target.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_preds = 1 - all_preds # 스케치(검정==0)와 배경(흰색==1) 을 역전
    all_targets = 1 - all_targets # 스케치(검정==0)와 배경(흰색==1) 을 역전

    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    wandb.log({
        "test_precision": precision,
        "test_recall"   : recall,
        "test_f1score"   : f1,
    })

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train, validate, and test a model.")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--train_yaml', type=str, required=True, help='Path to the training dataset YAML file')
    parser.add_argument('--val_yaml', type=str, required=True, help='Path to the validation dataset YAML file')
    parser.add_argument('--test_yaml', type=str, required=True, help='Path to the test dataset YAML file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=16, help="num workers for dataloader")
    parser.add_argument('--model_name', type=str, default="FPathPredictor", choices=["FPathPredictor", "UNetFPathPredictor"])

    args = parser.parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="VTFPredictor",

        # track hyperparameters and run metadata
        config=vars(args)
    )

    print(f"============ Parameters ==================")
    for k, v in vars(args).items():
        print(f"[{k:20s}]: {v}")
    print(f"==========================================")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"results_{timestamp}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args=args, mode=args.model_name)

    if args.model_name == "FPathPredictor":
        model = FPathPredictor()
    elif args.model_name == "UNetFPathPredictor":
        model = UNetFPathPredictor()
        
    objective = SketchMaskLoss()
    optimizer = optim.AdamW(list(model.parameters()) + list(objective.parameters()), lr=args.learning_rate)
    # objective = FocalLoss(alpha=10000, gamma=100).to('cuda') # gamma = focusing factor, easy negative case에 대하여 더 큰 패널티를 줌.
    # objective = nn.BCEWithLogitsLoss().to('cuda')

    model_selector = BestModelSelector(model=model, args=args, metric="loss", mode="min")

    accelerator = Accelerator()
    model, train_loader, val_loader, test_loader = accelerator.prepare(model, train_loader, val_loader, test_loader)
    for epoch in range(args.epochs):
        train(model, optimizer, objective, train_loader, accelerator)
        val(model, objective, val_loader, epoch, accelerator, model_selector)
        if (epoch+1) % 5 == 0:
            test(model, test_loader, accelerator)        
        
    test(model, test_loader, accelerator)

if __name__ == "__main__":
    main()
