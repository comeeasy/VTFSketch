import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import wandb
import lightning as L
from sklearn.metrics import precision_score, recall_score, f1_score

from src.losses import SketchMaskLoss, SketchNoiseMaskLoss



# Training, Validation, and Test Functions
class VTFPredictor(L.LightningModule):
    def __init__(self, model_name, learning_rate, batch_size, num_workers, loss_name):
        super(VTFPredictor, self).__init__()
        self.save_hyperparameters()
        if model_name == "FPathPredictor":
            self.model = FPathPredictor()
        elif model_name == "UNetFPathPredictor":
            self.model = UNetFPathPredictor()
        
        self.loss_name = loss_name
        if loss_name == "SketchMaskLoss":
            self.objective = SketchMaskLoss()
        elif loss_name == "SketchNoiseMaskLoss":
            self.objective = SketchNoiseMaskLoss()
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, vtf, img):
        return self.model(vtf, img)

    def inference(self, vtf, img):
        return torch.tensor(self(vtf, img).clone().detach() > 0.5, dtype=torch.float32)

    def configure_optimizers(self):
        return optim.AdamW(list(self.model.parameters()) + list(self.objective.parameters()), lr=self.learning_rate)

    def calculate_loss(self, pred_target, target, infodraw):
        if self.loss_name == "SketchMaskLoss":
            loss = self.objective(pred_target, target, None)
        elif self.loss_name == "SketchNoiseMaskLoss":
            loss = self.objective(pred_target, target, infodraw) # 10번째 channel에 infodraw가 저장
        else:
            raise RuntimeError("Spcify correct loss. in [\"SketchMaskLoss\", \"SketchNoiseMaskLoss\"]")
        return loss

    def training_step(self, batch, batch_idx):
        vtf, img, infodraw, target = batch
        pred_target = self(vtf, img)
        
        loss = self.calculate_loss(pred_target=pred_target, target=target, infodraw=infodraw)
        
        self.log("train_loss", loss, sync_dist=True)
        # print(f"train_loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        vtf, img, infodraw, target = batch
        pred_target = self.model(vtf, img)
        loss = self.calculate_loss(pred_target=pred_target, target=target, infodraw=infodraw)
        self.log("val_loss", loss, sync_dist=True)
        # print(f"val_loss: {loss}")
        # pred_target = self.inference(vtf, img)
        images = wandb.Image((pred_target > 0.5).float(), caption="val results")
        self.logger.experiment.log({"val_results": images})
        self.validation_step_outputs.append({"preds": pred_target, "targets": target})

    def test_step(self, batch, batch_idx):
        vtf, img, infodraw, target = batch
        # pred_target = self.inference(vtf, img)
        pred_target = self(vtf, img)
        self.test_step_outputs.append({"preds": pred_target, "targets": target})

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        
        all_preds = np.concatenate([x["preds"].cpu().numpy() > 0.5 for x in outputs]).flatten()
        all_targets = np.concatenate([x["targets"].cpu().numpy() > 0.5 for x in outputs]).flatten()
        all_preds = 1 - all_preds
        all_targets = 1 - all_targets
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        self.log("val_precision", precision, sync_dist=True)
        self.log("val_recall", recall, sync_dist=True)
        self.log("val_f1score", f1, sync_dist=True)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        
        all_preds = np.concatenate([x["preds"].cpu().numpy() for x in outputs]).flatten()
        all_targets = np.concatenate([x["targets"].cpu().numpy() for x in outputs]).flatten()
        all_preds = 1 - all_preds
        all_targets = 1 - all_targets
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        self.log("test_precision", precision, sync_dist=True)
        self.log("test_recall", recall, sync_dist=True)
        self.log("test_f1score", f1, sync_dist=True)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        self.test_step_outputs.clear()


class ResFCLayer(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False, dropout_p=0.0) -> None:
        super().__init__()
        
        self.is_res_con = in_channel == out_channel
        self.hidden_channel = in_channel // 4 if in_channel > 32 else in_channel
        
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
        
    def forward(self, vtf, img):
        x = vtf.permute((0, 2, 3, 1)) # [B x C x H x W] -> [B x H x W x C]
        out = self.layer1(x)
        out = self.final_layer(out)
        
        out = out.permute((0, 3, 1, 2)) # [B x H x W x C] -> [B x C x H x W]
        return torch.sigmoid(out)

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
    def __init__(self, in_channels=24, out_channels=1, init_features=64, num_layers=5, dropout_p=0.0):
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

    def forward(self, vtf, img):
        x = torch.cat([vtf, img], dim=1)
        enc_features = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)
            x = self.pool(x)

        # Bottleneck
        bottleneck = enc_features[-1]
        enc_features = enc_features[:-1][::-1]

        # Decoder path
        for idx, (up_conv, decoder) in enumerate(zip(self.up_convs, self.decoders)):
            bottleneck = up_conv(bottleneck)
            bottleneck = torch.cat((bottleneck, enc_features[idx]), dim=1)
            bottleneck = decoder(bottleneck)

        return torch.sigmoid(self.final_conv(bottleneck))

    def _block(self, in_channels, features, kernel_size, stride, padding, dropout_p):
        return nn.Sequential(
            NaiveCNNBlock(in_channels=in_channels, out_channels=features, kernel_size=kernel_size, stride=stride, padding=padding, dropout_p=dropout_p),
            NaiveCNNBlock(in_channels=features, out_channels=features, kernel_size=kernel_size, stride=stride, padding=padding, dropout_p=dropout_p)
        )