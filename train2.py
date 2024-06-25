import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from src.models import ResFCLayer



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
            nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1, bias=bias),
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
    dset = FPathDataset(data_path="dataset/train_small.npz")
    val_dset = FPathDataset(data_path="dataset/val.npz")
    dloader = DataLoader(dset, batch_size=40960, num_workers=24, drop_last=True)
    val_dloader = DataLoader(val_dset, batch_size=40960, num_workers=24)

    model = FCModel()
    # model = CNNModel()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    objective = nn.BCELoss()
    
    model = model.to("cuda")
    objective = objective.to('cuda')

    for epoch in range(10):
        model = model.train()
        sum_loss = 0
        for fpath, target in tqdm(dloader):
            fpath, target = fpath.to('cuda'), target.to('cuda')
            y_hat = model(fpath)
            
            loss = objective(y_hat.squeeze(1), target)
            sum_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model = model.eval()
        sTP, sFP, sTN, sFN = 0, 0, 0, 0
        max_accuracy = 0
        with torch.no_grad():
            for fpath, target in tqdm(val_dloader):
                fpath, target = fpath.to('cuda'), target.to('cuda')
                y_hat = model(fpath)

                torch.sum((y_hat.squeeze() > 0.5) == target)

                sTP += torch.sum(((y_hat.squeeze() > 0.5) == target) * (target == 1))
                sFP += torch.sum(((y_hat.squeeze() > 0.5) == target) * (target == 0))
                sTN += torch.sum(((y_hat.squeeze() < 0.5) == target) * (target == 0))
                sFN += torch.sum(((y_hat.squeeze() < 0.5) == target) * (target == 1))
        
        accuracy = (sTP + sTN) / (sTP + sFP + sTN + sFN)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(model, f"weights/best_FCModel_acc{accuracy*100:2.2f}.pth")
            
        print(f"epoch: {epoch} || val accuracy: {accuracy}, step_loss: {sum_loss / len(dloader)}")
        print(f"\tTP: {sTP}, FN: {sFN}")
        print(f"\tFP: {sFP}, TN: {sTN}")
    