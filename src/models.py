import torch
import torch.nn as nn
import torch.nn.init as init



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
            ResFCLayer(128, 128, dropout_p=0.5, bias=False),
        )
        
        self.final_layer = nn.Sequential(
            nn.Linear(128, out_channel, bias=True)
        )
        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.final_layer(out)
        
        return out

class NaiveCNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dropout_p=0.0):
        super(NaiveCNNBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
    
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
    def __init__(self, in_channel=21 + 3, out_channel=1):
        super().__init__()
        
        self.encoder1 = nn.Sequential(
            NaiveCNNBlock(in_channel=in_channel, out_channel=64, kernel_size=21, stride=1, padding=10),
            NaiveCNNBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            NaiveCNNBlock(in_channel=64, out_channel=128, kernel_size=3, stride=1, padding=1),
            NaiveCNNBlock(in_channel=128, out_channel=128, kernel_size=3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            NaiveCNNBlock(in_channel=128, out_channel=256, kernel_size=3, stride=1, padding=1),
            NaiveCNNBlock(in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            NaiveCNNBlock(in_channel=256, out_channel=512, kernel_size=3, stride=1, padding=1),
            NaiveCNNBlock(in_channel=512, out_channel=512, kernel_size=3, stride=1, padding=1),
        )
        self.encoder5 = nn.Sequential(
            NaiveCNNBlock(in_channel=512, out_channel=1024, kernel_size=3, stride=1, padding=1),
            NaiveCNNBlock(in_channel=1024, out_channel=1024, kernel_size=3, stride=1, padding=1),
        )
        self.decoder1 = nn.Sequential(
            NaiveCNNBlock(in_channel=128, out_channel=64, kernel_size=3, stride=1, padding=1),
            NaiveCNNBlock(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
        )
        self.decoder2 = nn.Sequential(
            NaiveCNNBlock(in_channel=256, out_channel=128, kernel_size=3, stride=1, padding=1),
            NaiveCNNBlock(in_channel=128, out_channel=128, kernel_size=3, stride=1, padding=1),
        )
        self.decoder3 = nn.Sequential(
            NaiveCNNBlock(in_channel=512, out_channel=256, kernel_size=3, stride=1, padding=1),
            NaiveCNNBlock(in_channel=256, out_channel=256, kernel_size=3, stride=1, padding=1),
        )
        self.decoder4 = nn.Sequential(
            NaiveCNNBlock(in_channel=1024, out_channel=512, kernel_size=3, stride=1, padding=1),
            NaiveCNNBlock(in_channel=512, out_channel=512, kernel_size=3, stride=1, padding=1),
        )
        self.up_conv21 = NaiveTransposedCNNBlock(in_channel=128, out_channel=64)
        self.up_conv32 = NaiveTransposedCNNBlock(in_channel=256, out_channel=128)
        self.up_conv43 = NaiveTransposedCNNBlock(in_channel=512, out_channel=256)
        self.up_conv54 = NaiveTransposedCNNBlock(in_channel=1024, out_channel=512)
        self.down_sample12 = nn.MaxPool2d(2, 2)
        self.down_sample23 = nn.MaxPool2d(2, 2)
        self.down_sample34 = nn.MaxPool2d(2, 2)
        self.down_sample45 = nn.MaxPool2d(2, 2)
        
    def forward(self, vtf, img):
        x = torch.concat([vtf, img], dim=1)
        
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.down_sample12(x1))
        x3 = self.encoder3(self.down_sample23(x2))
        x4 = self.encoder4(self.down_sample34(x3))
        x5 = self.encoder5(self.down_sample45(x4))
        
        u4 = torch.concat([x4, self.up_conv54(x5)], dim=1)
        u4 = self.decoder4(u4)
        u3 = torch.concat([x3, self.up_conv43(u4)], dim=1)
        u3 = self.decoder3(u3)
        u2 = torch.concat([x2, self.up_conv32(u3)], dim=1)
        u2 = self.decoder2(u2)
        u1 = torch.concat([x1, self.up_conv21(u2)], dim=1)
        
        out = self.decoder1(u1)
        
        return out