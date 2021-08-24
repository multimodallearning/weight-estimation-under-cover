from models.bpsfc import BPSMLP
from models.cnn3d import CNN3D
from models.pointnet import PointNet
from models.unet3d_cnn3d_e2e import UNet3D_CNN3D


def create_model(cfg):
    arch = cfg.MODEL.ARCHITECTURE
    if arch == 'PointNet':
        model = PointNet()
    elif arch == 'BPS':
        model = BPSMLP(cfg)
    elif arch == 'CNN3D':
        model = CNN3D(cfg)
    elif arch == 'UNet_CNN3D':
        model = UNet3D_CNN3D(cfg)
    else:
        raise ValueError
    return model
