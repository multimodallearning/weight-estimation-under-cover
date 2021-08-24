import sys
sys.path.append('../')
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.unet3d import UNet3D
from models.cnn3d import CNN3D
from data.slp_weight_dataset import SLPWeightDataset
from configs.defaults import get_cfg_defaults


def test(cfg, unet_path, cnn3d_path):
    # DATA
    train_set = SLPWeightDataset(cfg, phase='train')
    val_set = SLPWeightDataset(cfg, phase='val')
    val_loader = DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE_TEST, num_workers=cfg.INPUT.NUM_WORKERS,
                            shuffle=False, drop_last=False)

    # MODEL
    model_unet = UNet3D().to(cfg.MODEL.DEVICE)
    model_unet.load_state_dict(torch.load(unet_path))
    model_cnn3d = CNN3D(cfg).to(cfg.MODEL.DEVICE)
    model_cnn3d.load_state_dict(torch.load(cnn3d_path))
    model_unet.eval()
    model_cnn3d.eval()

    val_loss = 0.
    for it, data in enumerate(val_loader, 1):
        inputs, labels, idx = data
        inputs = inputs.to(cfg.MODEL.DEVICE)
        labels = labels.to(cfg.MODEL.DEVICE)

        with torch.no_grad():
            uncov_vol = model_unet(inputs)
            # convert prediction into binary grid and mean-center the non-zero elements inside the grid before
            # further processing  by 3D CNN
            uncov_vol = F.softmax(uncov_vol, dim=1)[:, 1, :, :, :]
            uncov_vol = uncov_vol.cpu().numpy()
            mc_vol = np.zeros_like(uncov_vol)
            for i, vol in enumerate(uncov_vol):
                vol = np.float32(vol > 0.5)

                x, y, z = np.where(vol == 1)
                mc_x = np.clip((x - (np.mean(x) - int(train_set.grid_shape[0] / 2))).astype(int), 0,
                                train_set.grid_shape[0] - 1)
                mc_y = np.clip((y - (np.mean(y) - int(train_set.grid_shape[1] / 2))).astype(int), 0,
                                train_set.grid_shape[1] - 1)
                mc_z = np.clip((z - (np.mean(z) - int(train_set.grid_shape[2] / 2))).astype(int), 0,
                                train_set.grid_shape[2] - 1)
                vol = np.zeros_like(vol)
                vol[mc_x, mc_y, mc_z] = 1.

                vol = np.expand_dims(vol, axis=0)
                mc_vol[i, :, :, :] = vol
            uncov_vol = torch.from_numpy(mc_vol).to(cfg.MODEL.DEVICE)
            uncov_vol = uncov_vol.unsqueeze(1)

            weight_pred = model_cnn3d(uncov_vol)

        if cfg.INPUT.NORMALIZE_OUTPUT:
            weight_pred = weight_pred * (train_set.max_weight_train - train_set.min_weight_train) + train_set.min_weight_train
            labels = labels * (train_set.max_weight_train - train_set.min_weight_train) + train_set.min_weight_train

        val_loss += torch.sum(torch.abs(weight_pred.view(-1) - labels.view(-1))).item()

    val_loss = val_loss / len(val_set)
    print('MAE: ', val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default="0",
        metavar="FILE",
        help="gpu to train on",
        type=str,
    )
    parser.add_argument(
        "--unet-path",
        default="",
        metavar="FILE",
        help="path to pre-trained u-net model",
        type=str,
    )
    parser.add_argument(
        "--cnn3d-path",
        default="",
        metavar="FILE",
        help="path to pre-trained 3D CNN",
        type=str,
    )
    parser.add_argument(
        "--val-split",
        default="dana",
        metavar="FILE",
        help="whether to evaluate on danaLab or simLab data",
        type=str,
    )
    parser.add_argument(
        "--cover-condition",
        default="uncover",
        metavar="FILE",
        help="The cover condition to evaluate on. Should be in {uncover, cover1, cover2, cover12}",
        type=str,
    )
    parser.add_argument(
        "--position",
        default="supine",
        metavar="FILE",
        help="The position to evaluate on. Should be in {all, lateral, supine, left, right}",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    opts = ["SLP_DATASET.VAL_SPLIT", args.val_split, "SLP_DATASET.COVER_CONDITION", args.cover_condition,
            "SLP_DATASET.POSITION", args.position]
    cfg.merge_from_list(opts)
    cfg.freeze()

    if args.unet_path == "":
        working_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
        unet_path = os.path.join(working_directory, 'model_unet.pth')
    else:
        unet_path = args.unet_path
    if not os.path.isfile(unet_path):
        raise ValueError('There is no pre-trained U-Net at the specified path. Set the model path correctly or'
                         ' run training first.')

    if args.cnn3d_path == "":
        working_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
        cnn3d_path = os.path.join(working_directory, 'model_cnn3d.pth')
    else:
        cnn3d_path = args.cnn3d_path
    if not os.path.isfile(cnn3d_path):
        raise ValueError('There is no pre-trained 3D CNN at the specified path. Set the model path correctly or'
                         ' run training first.')

    test(cfg, unet_path, cnn3d_path)