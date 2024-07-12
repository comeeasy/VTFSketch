import numpy as np

import wandb
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from src.models import ResFCLayer
from src.utils import calculate_noise_metric



class FPathDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data = np.load(data_path)["arr_0"] 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        fpath = self.data[index][:21]
        target = self.data[index][-1]
        return fpath, target

class FCModel(nn.Module):
    def __init__(self, in_channel=21, out_channel=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer1 = nn.Sequential(
            ResFCLayer(in_channel, 128, bias=True),
            ResFCLayer(128, 128, bias=True),
            ResFCLayer(128, 128, bias=True)
        )
        self.final_layer = nn.Sequential(
            ResFCLayer(128, 128, bias=True, dropout_p=0.5),
            nn.Linear(128, 1, bias=True)
        )
        
    def forward(self, fpath_list_tensor):
        out = self.layer1(fpath_list_tensor)
        out = self.final_layer(out)
        return torch.sigmoid(out)

class ResCNN1DLayer(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False, 
                        dropout_p=0.0) -> None:
        super().__init__()
        self.is_res_con = in_channel == out_channel
        self.layer = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=5, padding=2, bias=bias),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self._init_layer()
    
    def forward(self, x):
        if self.is_res_con:
            out = x + self.layer(x)
        else:
            out = self.layer(x)
        return out
        
    def _init_layer(self):
        for m in self.layer.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class CNNModel(nn.Module):
    def __init__(self, in_channel=21, out_channel=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer1 = nn.Sequential(
            ResCNN1DLayer(1, 128, bias=True),
            ResCNN1DLayer(128, 128, bias=True),
            ResCNN1DLayer(128, 128, bias=True)
        )
        self.final_layer = nn.Sequential(
            ResFCLayer(128 * 21, 128, bias=True, dropout_p=0.5),
            nn.Linear(128, 1, bias=True)
        )
        
    def forward(self, fpath_list_tensor):
        out = fpath_list_tensor.unsqueeze(1)
        out = self.layer1(out)
        out = torch.flatten(out, start_dim=1)
        out = self.final_layer(out)
        return torch.sigmoid(out)


if __name__ == "__main__":
    wandb.login(relogin=True)
    wandb.init(project="VTFSketch")
    
    dset = FPathDataset(data_path="dataset/train_small.npz")
    val_dset = FPathDataset(data_path="dataset/val.npz")
    dloader = DataLoader(dset, batch_size=4096, num_workers=24, drop_last=True)
    val_dloader = DataLoader(val_dset, batch_size=4096, num_workers=24)

    # model = FCModel()
    model = CNNModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    objective = nn.BCELoss()
    
    device = "cuda:1"
    
    model = model.to(device)
    objective = objective.to(device)

    for epoch in range(1000):
        model = model.train()
        train_sum_loss = 0
        for fpath, target in tqdm(dloader):
            fpath, target = fpath.to(device), target.to(device)
            y_hat = model(fpath)
            
            loss = objective(y_hat.squeeze(1), target)
            train_sum_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model = model.eval()
        max_accuracy = 0
        preds_list, targets_list = [], []
        val_sum_loss = 0
        with torch.no_grad():
            for fpath, target in tqdm(val_dloader):
                fpath, target = fpath.to(device), target.to(device)
                y_hat = model(fpath)

                loss = objective(y_hat.squeeze(1), target)
                val_sum_loss += loss.item()

                targets_list.append(target.detach().cpu().numpy())
                preds_list.append(y_hat.detach().cpu().numpy())

        preds = np.concatenate(preds_list).flatten()
        targets = np.concatenate(targets_list).flatten()
        metric = calculate_noise_metric(preds=preds, targets=targets)

        if metric['accuracy'] > max_accuracy:
            max_accuracy = metric['accuracy']
            torch.save(model, f"weights/best_CNNModel_acc{max_accuracy*100:2.2f}.pth")
        
        print(f"epoch: {epoch} || val accuracy: {metric['accuracy']}, val f1score: {metric['f1score']}, val recall: {metric['recall']}  val_loss: {val_sum_loss / len(val_dloader)}")
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_sum_loss / len(dloader),
            "val_loss": val_sum_loss / len(val_dloader),
            "val_accuracy": metric['accuracy'],
            "val_f1score": metric['f1score'],
            "val_recall": metric['recall']
        })
    
    wandb.finish()
    