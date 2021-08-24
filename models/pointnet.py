import torch
import torch.nn as nn


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.local_layers = nn.ModuleList([
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ])

        self.global_layers = nn.ModuleList([
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        ])

        self.output_layers = nn.ModuleList([
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        ])

    def forward(self, x):
        for layer in self.local_layers:
            x = layer(x)
        for layer in self.global_layers:
            x = layer(x)
        x = torch.max(x, dim=2, keepdim=False)[0]
        for layer in self.output_layers:
            x = layer(x)

        return x
