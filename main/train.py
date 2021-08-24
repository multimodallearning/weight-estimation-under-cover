import sys
sys.path.append('../')
import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from models.unet3d import UNet3D
from models.cnn3d import CNN3D
from configs.defaults import get_cfg_defaults
from data.slp_weight_dataset import SLPWeightDataset
from data.slp_uncover_dataset import SLPUncoverDataset


def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label).fill_(0)
    for label_num in range(0, max_label):
        iflat = (outputs == label_num).view(-1).float()
        tflat = (labels == label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice


def train_uncovering(cfg, output_directory):

    train_set = SLPUncoverDataset(cfg, phase='train')
    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN, num_workers=cfg.INPUT.NUM_WORKERS,
                              shuffle=True)
    val_set = SLPUncoverDataset(cfg, phase='val')
    val_loader = DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE_TEST, num_workers=cfg.INPUT.NUM_WORKERS,
                            shuffle=False)

    # model
    model = UNet3D().to(cfg.MODEL.DEVICE)

    # optimizer and scheduler
    use_amp = cfg.SOLVER.USE_AMP  # whether to use mixed precision
    optimizer = Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lr_scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_MILESTONES, cfg.SOLVER.LR_GAMMA)

    # loss criterion
    criterion = nn.CrossEntropyLoss().to(cfg.MODEL.DEVICE)

    # logging: save training loss, validation loss and validation accuracy after each epoch to numpy array
    validation_log = np.zeros([cfg.SOLVER.NUM_EPOCHS, 3])

    # Training
    for e in range(cfg.SOLVER.NUM_EPOCHS):
        model.train()
        start_time = time.time()
        loss_values = []

        for it, data in enumerate(train_loader, 1):
            inputs, labels, idx = data
            inputs = inputs.to(cfg.MODEL.DEVICE)
            labels = labels.to(cfg.MODEL.DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(inputs)
                loss = criterion(pred, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_values.append(loss.item())

        lr_scheduler.step()

        # Validation
        model.eval()

        train_loss = np.mean(loss_values)

        loss_values = []
        all_dice = []

        for it, data in enumerate(val_loader, 1):
            inputs, labels, idx = data
            inputs = inputs.to(cfg.MODEL.DEVICE)
            labels = labels.to(cfg.MODEL.DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    pred = model(inputs)

                loss = criterion(pred, labels)
            loss_values.append(loss.item())

            pred = pred.argmax(dim=1)
            dice = dice_coeff(pred, labels, 2)
            all_dice.append(dice)

        val_loss = np.mean(loss_values)
        all_dice = torch.stack(all_dice, dim=0)

        end_time = time.time()

        print('epoch', e, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss', '%0.3f' % train_loss,
              'val_loss', '%0.3f' % val_loss, 'val_dice', torch.mean(all_dice, dim=0))

        validation_log[e, :] = [train_loss, val_loss, torch.mean(all_dice, dim=0)[1].item()]
        np.save(os.path.join(output_directory, "uncovering_log"), validation_log)

        torch.save(model.state_dict(), os.path.join(output_directory, 'model_unet.pth'))


def train_weight_regression(cfg, output_directory):

    train_set = SLPWeightDataset(cfg, phase='train')
    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN, num_workers=cfg.INPUT.NUM_WORKERS,
                              shuffle=True)
    val_set = SLPWeightDataset(cfg, phase='val')
    val_loader = DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE_TEST, num_workers=cfg.INPUT.NUM_WORKERS,
                            shuffle=False)

    # model
    model_unet = UNet3D().to(cfg.MODEL.DEVICE)
    unet_path = os.path.join(output_directory, 'model_unet.pth')
    if not os.path.isfile(unet_path):
        raise ValueError('The given path to the U-Net does not exist. Please train uncovering first.')
    model_unet.load_state_dict(torch.load(unet_path))
    model_unet.eval()
    model_cnn3d = CNN3D(cfg).to(cfg.MODEL.DEVICE)

    # optimizer and scheduler
    use_amp = cfg.SOLVER.USE_AMP
    optimizer = Adam(list(model_cnn3d.parameters()), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lr_scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_MILESTONES, cfg.SOLVER.LR_GAMMA)

    # loss criterion
    criterion = nn.MSELoss(reduction='mean')

    # Training
    validation_log = np.zeros([cfg.SOLVER.NUM_EPOCHS, 2])
    for e in range(cfg.SOLVER.NUM_EPOCHS):
        model_cnn3d.train()
        start_time = time.time()
        loss_values = []

        for it, data in enumerate(train_loader, 1):
            inputs, labels, idx = data
            inputs = inputs.to(cfg.MODEL.DEVICE)
            labels = labels.to(cfg.MODEL.DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    uncov_vol = model_unet(inputs)

                # convert prediction into binary grid and mean-center the non-zero elements inside the grid before
                # further processing  by 3D CNN
                uncov_vol = F.softmax(uncov_vol, dim=1)[:, 1, :, :, :]
                # TODO: implement mean-centering in PyTorch to remain on GPU
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
                loss = criterion(weight_pred.view(-1), labels.view(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_values.append(loss.item())

        lr_scheduler.step()

        # Validation
        model_cnn3d.eval()

        train_loss = np.mean(loss_values)
        val_loss = 0.

        for it, data in enumerate(val_loader, 1):
            inputs, labels, idx = data
            inputs = inputs.to(cfg.MODEL.DEVICE)
            labels = labels.to(cfg.MODEL.DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
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

        end_time = time.time()
        print('epoch', e, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss', '%0.3f' % train_loss,
              'val_loss', '%0.3f' % val_loss,)

        validation_log[e, :] = [train_loss, val_loss]
        np.save(os.path.join(output_directory, "weight_log"), validation_log)

        torch.save(model_cnn3d.state_dict(), os.path.join(output_directory, 'model_cnn3d.pth'))


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
        "--stage",
        default="uncovering",
        metavar="FILE",
        help="Whether to learn uncovering or weight regression. Should be in {uncovering, weight}.",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.stage == 'uncovering':
        train_uncovering(cfg, output_directory)
    elif args.stage == 'weight':
        train_weight_regression(cfg, output_directory)
    else:
        raise ValueError('Stage should be uncovering or weight.')