import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn3d import CNN3D


class UNet3D_CNN3D(nn.Module):
    def __init__(self, cfg):
        super(UNet3D_CNN3D, self).__init__()

        self.down1a = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU())
        self.down1b = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU())
        self.max1 = nn.MaxPool3d(2)

        self.down2a = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU())
        self.down2b = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU())
        self.max2 = nn.MaxPool3d(2)

        self.down3a = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU())
        self.down3b = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(256),
                                    nn.ReLU())
        self.max3 = nn.MaxPool3d(2)

        self.down4a = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(256),
                                    nn.ReLU())
        self.down4b = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(512),
                                    nn.ReLU())

        self.up1 = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=2, stride=2)

        self.trans1a = nn.Sequential(nn.Conv3d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(256),
                                    nn.ReLU())
        self.trans1b = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(256),
                                     nn.ReLU())

        self.trans2a = nn.Sequential(nn.Conv3d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU())
        self.trans2b = nn.Sequential(nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm3d(128),
                                     nn.ReLU())

        self.trans3a = nn.Sequential(nn.Conv3d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(),
                                    nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(),
                                    nn.Conv3d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0))

        self.predictor = CNN3D(cfg)

    def forward(self, x):

        x = self.down1a(x)
        x = self.down1b(x)
        x1 = x
        x = self.max1(x)
        x = self.down2a(x)
        x = self.down2b(x)
        x2 = x
        x = self.max2(x)
        x = self.down3a(x)
        x = self.down3b(x)
        x3 = x
        x = self.max3(x)
        x = self.down4a(x)
        x = self.down4b(x)
        x = self.up1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.trans1a(x)
        x = self.trans1b(x)
        x = self.up2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.trans2a(x)
        x = self.trans2b(x)
        x = self.up3(x)
        x = torch.cat((x, x1), dim=1)
        x = self.trans3a(x)

        x = F.softmax(x, dim=1)[:, 1:, :, :, :]

        x = self.predictor(x)
        return x
