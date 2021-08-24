import torch
import torch.nn as nn


class CNN3D(nn.Module):
    def __init__(self, cfg):
        super(CNN3D, self).__init__()

        self.features = nn.ModuleList([
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)
        ])

        # compute the size of the flattened features that will be forwarded by FC layers
        input_shape = tuple(cfg.INPUT.VOXEL_GRID_SHAPE)
        x = torch.autograd.Variable(torch.rand((1, 1) + input_shape))
        for f in self.features:
            x = f(x)
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.fc = nn.ModuleList([
            nn.Linear(in_features=dim_feat, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1)
        ])

    def forward(self, x):

        for layer in self.features:
            x = layer(x)
        x = torch.flatten(x, 1)
        for layer in self.fc:
            x = layer(x)

        return x
