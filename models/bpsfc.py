import numpy as np
import torch
import torch.nn as nn
from bps import bps


def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)


class BPSMLP(nn.Module):
    def __init__(self, cfg):
        super(BPSMLP, self).__init__()
        self.basis = bps.generate_random_basis(cfg.MODEL.BPS.NUM_BASIS_POINTS, n_dims=3, radius=cfg.MODEL.BPS.RADIUS,
                                                   random_seed=cfg.MODEL.BPS.SEED)
        self.basis = torch.from_numpy(np.float32(self.basis)).to(cfg.MODEL.DEVICE)

        n_features = cfg.MODEL.BPS.NUM_BASIS_POINTS
        self.layers = nn.ModuleList([
            nn.BatchNorm1d(n_features),
            nn.Linear(in_features=n_features, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=1024, out_features=1)
        ])

    def forward(self, x):
        # compute BPS encoding
        B, D, N = x.size()
        x = x.transpose(1, 2)
        x = x.reshape(-1, D)
        local_dist_matrix = pairwise_distances(x, self.basis)
        local_dist_matrix = local_dist_matrix.view(B, N, self.basis.size(0))
        x = torch.sqrt(torch.min(local_dist_matrix, dim=1)[0])

        for layer in self.layers:
            x = layer(x)

        return x