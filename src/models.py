import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchmetrics

import wandb
import lightning as L
from sklearn.metrics import precision_score, recall_score, f1_score

from src.losses import SketchMaskLoss, SketchNoiseMaskLoss, MaskedBCELoss
from src.unet.unet_models import UNet

from tqdm import tqdm




# Training, Validation, and Test Functions
class VTFPredictor(L.LightningModule):
    def __init__(self, model_name, learning_rate=1e-4, batch_size=1, num_workers=-1, 
                 loss_name="SketchMaskLoss", inference_mode="Selective"):
        super(VTFPredictor, self).__init__()
        if model_name == "FPathPredictor":
            self.model = FPathPredictor()
        elif model_name == "UNetFPathPredictor":
            self.model = UNet(in_channels=24, out_channels=1) # [vtf + img]  = 24 x H x W
        elif model_name == "UNetFPathPredictor_VTFOnly":
            self.model = UNet(in_channels=21, out_channels=1) # vtf  = 21 x H x W
        elif model_name == "MinFPathPredictor":
            self.model = MinFPathPredictor()
        
        self.loss_name = loss_name
        self.threshold_W = 0.99
        if loss_name == "SketchMaskLoss":
            self.objective = SketchMaskLoss()
        elif loss_name == "SketchNoiseMaskLoss":
            self.objective = SketchNoiseMaskLoss(threshold_W=self.threshold_W)
        elif loss_name == "MaskedBCELoss":
            self.objective = MaskedBCELoss()

        self.inference_mode = inference_mode

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.f1score    = torchmetrics.F1Score(task='binary')
        self.recall     = torchmetrics.Recall(task='binary')
        self.precision  = torchmetrics.Precision(task='binary')

        self.save_hyperparameters()

    def forward(self, vtf, img, infodraw):
        return self.model(vtf, img, infodraw)

    def inference(self, vtf, img, infodraw):
        pred = self(vtf=vtf, img=img, infodraw=infodraw)
        
        if self.inference_mode == "Selective":
            mask = (infodraw < self.threshold_W).float()
            masked_pred = (mask * pred) + (1-mask * infodraw)
            result = torch.tensor(masked_pred > 0.5).float().detach()
        elif self.inference_mode == "Whole":
            masked_pred = pred.clone()
            result = torch.tensor(masked_pred > 0.5).float().detach()  
        else:
            raise RuntimeError(f"inference mode is {self.inference_mode}, chech usage on train_lightning.py")
        
        return pred, masked_pred, result

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def calculate_loss(self, pred_target, target, infodraw):
        if self.loss_name == "SketchMaskLoss":
            loss = self.objective(pred_target, target, None)
        elif self.loss_name == "SketchNoiseMaskLoss":
            loss = self.objective(pred_target, target, infodraw)
        elif self.loss_name == "MaskedBCELoss":
            mask = infodraw < self.threshold_W
            loss = self.objective(pred=pred_target, target=target, mask=mask)
        else:
            raise RuntimeError("Spcify correct loss. in [\"SketchMaskLoss\", \"SketchNoiseMaskLoss\"]")
        return loss

    def training_step(self, batch, batch_idx):
        vtf, img, infodraw, target = batch
        pred_target = self(vtf=vtf, img=img, infodraw=infodraw)
        
        loss = self.calculate_loss(pred_target=pred_target, target=target, infodraw=infodraw)
        
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vtf, img, infodraw, target = batch
        pred_target, masked_pred, result = self.inference(vtf=vtf, img=img, infodraw=infodraw)
        loss = self.calculate_loss(pred_target=pred_target, target=target, infodraw=infodraw)
        self.log("val_loss", loss, sync_dist=True)
        
        results = wandb.Image(result, caption="val results")
        self.logger.experiment.log({"val_results": results})
        masked_preds = wandb.Image(masked_pred)
        self.logger.experiment.log({"val_masked_pred": masked_preds})
        
        # 하얀 배경이 아닌 스케치를 정답으로 삼음
        self.f1score.update(1-result, 1-target) 
        self.recall.update(1-result, 1-target)
        self.precision.update(1-result, 1-target)

    def test_step(self, batch, batch_idx):
        vtf, img, infodraw, target = batch
        _, _, result = self.inference(vtf=vtf, img=img, infodraw=infodraw)

        self.f1score.update(1-result, 1-target) 
        self.recall.update(1-result, 1-target)
        self.precision.update(1-result, 1-target)

    def on_validation_epoch_end(self):
        # Compute the final metric values
        f1 = self.f1score.compute()
        recall = self.recall.compute()
        precision = self.precision.compute()

        self.log("val_precision", precision, sync_dist=True)
        self.log("val_recall", recall, sync_dist=True)
        self.log("val_f1score", f1, sync_dist=True)

        # Reset metrics after computation
        self.f1score.reset()
        self.recall.reset()
        self.precision.reset()

    def on_test_epoch_end(self):
        f1 = self.f1score.compute()
        recall = self.recall.compute()
        precision = self.precision.compute()

        self.log("test_precision", precision, sync_dist=True)
        self.log("test_recall", recall, sync_dist=True)
        self.log("test_f1score", f1, sync_dist=True)

        # Reset metrics after computation
        self.f1score.reset()
        self.recall.reset()
        self.precision.reset()


class ResFCLayer(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False, 
                        dropout_p=0.0) -> None:
        super().__init__()
        self.is_res_con = in_channel == out_channel
        self.layer = nn.Sequential(
            nn.Linear(in_channel, out_channel, bias=bias),
            nn.LayerNorm(out_channel),
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
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class FPathPredictor(nn.Module):
    def __init__(self, in_channel=21, out_channel=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer1 = nn.Sequential(
            ResFCLayer(in_channel, 128, bias=True),
            ResFCLayer(128, 128, bias=False),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(128, out_channel, bias=True)
        )
    def forward(self, vtf, img, infodraw):
        x = vtf.permute((0, 2, 3, 1)) # [B x C x H x W] -> [B x H x W x C]
        out = self.layer1(x)
        out = self.final_layer(out)
        
        out = out.permute((0, 3, 1, 2)) # [B x H x W x C] -> [B x C x H x W]
        return torch.sigmoid(out)
    
class MinFPathPredictor(nn.Module):
    def __init__(self, in_channel=21, out_channel=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer1 = nn.Sequential(
            ResFCLayer(in_channel, 128, bias=True),
            ResFCLayer(128, 128, bias=False),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(128, out_channel, bias=True)
        )
        
    def forward(self, vtf, img, infodraw):
        result = infodraw.clone()
        vtf_list = []
        B, _, H, W = infodraw.shape
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    if infodraw[b, :, h, w].item() < 0.99:
                        vtf_list.append((vtf[b, :, h, w], (h, w)))  

            for vtf_vec, (h, w) in vtf_list:
                out = self.layer1(vtf_vec.unsqueeze(0))
                out = self.final_layer(out)
                result[b, 0, h, w] = torch.sigmoid(out)
        
        return result
        

class NaiveCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_p=0.0):
        super(NaiveCNNBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
        self.__init_layer()
    
    def __init_layer(self):
        # Initialize the weights of the Conv2d layer
        for module in self.layer:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.layer(x)
    
class NaiveTransposedCNNBlock(nn.Module):
    # 2x2 up convolution
    
    def __init__(self, in_channel, out_channel, dropout_p=0.0):
        super(NaiveTransposedCNNBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
        self.__init_layer()
    
    def __init_layer(self):
        # Initialize the weights of the ConvTranspose2d layer
        for module in self.layer:
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        return self.layer(x)
    

class UNetFPathPredictor(nn.Module):
    def __init__(self, in_channels=21, out_channels=1, init_features=64, num_layers=5, dropout_p=0.0):
        super(UNetFPathPredictor, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Initialize the encoder blocks
        features = init_features
        for i in range(num_layers):
            self.encoders.append(
                self._block(
                        in_channels=in_channels if i == 0 else features // 2, 
                        features=features,
                        kernel_size=7 if i == 0 else 3,
                        stride=1,
                        padding=3 if i == 0 else 1,
                        dropout_p=dropout_p,
                )
            )
            features *= 2

        # Initialize the upsampling and decoder blocks
        for i in range(num_layers - 1, 0, -1):
            features //= 2
            self.up_convs.append(NaiveTransposedCNNBlock(features, features // 2, dropout_p=dropout_p))
            self.decoders.append(
                self._block(
                    in_channels=features, 
                    features=features // 2, 
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dropout_p=dropout_p
                )
            )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(features // 2, out_channels, kernel_size=1)

    def forward(self, vtf, img, infodraw):
        # x = torch.cat([vtf, img], dim=1)
        x = vtf
        enc_features = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            print(f"x: {x.shape}")
            enc_features.append(x)
            x = self.pool(x)

        # Bottleneck
        bottleneck = enc_features[-1]
        enc_features = enc_features[:-1][::-1]

        # Decoder path
        for idx, (up_conv, decoder) in enumerate(zip(self.up_convs, self.decoders)):
            bottleneck = up_conv(bottleneck)
            print(f"bottleneck: {bottleneck.shape}")
            bottleneck = torch.cat((bottleneck, enc_features[idx]), dim=1)
            bottleneck = decoder(bottleneck)

        return torch.sigmoid(self.final_conv(bottleneck))

    def _block(self, in_channels, features, kernel_size, stride, padding, dropout_p):
        return nn.Sequential(
            NaiveCNNBlock(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, stride=stride, padding=padding, dropout_p=dropout_p),
            NaiveCNNBlock(in_channels=features, out_channels=features, kernel_size=kernel_size, stride=stride, padding=padding, dropout_p=dropout_p)
        )