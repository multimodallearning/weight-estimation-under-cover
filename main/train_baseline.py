import sys
sys.path.append('../')
import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.utils import create_model
from configs.defaults import get_cfg_defaults
from data.slp_weight_dataset import SLPWeightDataset
from torch.optim.lr_scheduler import MultiStepLR



def train(cfg, output_directory):
    train_set = SLPWeightDataset(cfg, phase='train')
    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN, num_workers=cfg.INPUT.NUM_WORKERS,
                              shuffle=True)
    val_set = SLPWeightDataset(cfg, phase='val')
    val_loader = DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE_TEST, num_workers=cfg.INPUT.NUM_WORKERS,
                            shuffle=False)

    # model
    model = create_model(cfg).to(cfg.MODEL.DEVICE)

    # optimizer and scheduler
    use_amp = cfg.SOLVER.USE_AMP # whether to use mixed precision
    optimizer = Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lr_scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_MILESTONES, cfg.SOLVER.LR_GAMMA)

    # loss criterion
    criterion = nn.MSELoss(reduction='mean')

    # logging: save training and validation accuracy after each epoch to numpy array
    validation_log = np.zeros([cfg.SOLVER.NUM_EPOCHS, 2])

    # Training
    for e in range(cfg.SOLVER.NUM_EPOCHS):
        model.train()
        start_time = time.time()
        for it, data in enumerate(train_loader, 1):
            inputs, labels, idx = data
            inputs = inputs.to(cfg.MODEL.DEVICE)
            labels = labels.to(cfg.MODEL.DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(inputs)
                loss = criterion(pred.view(-1), labels.view(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        lr_scheduler.step()

        # Validation
        model.eval()

        # run evaluation on training set
        train_loss = 0.
        for it, data in enumerate(train_loader, 1):
            inputs, labels, idx = data
            inputs = inputs.to(cfg.MODEL.DEVICE)
            labels = labels.to(cfg.MODEL.DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    pred = model(inputs)

            # remap network output in range [0, 1] to weight in kg
            if cfg.INPUT.NORMALIZE_OUTPUT:
                pred = pred * (train_set.max_weight_train - train_set.min_weight_train) + train_set.min_weight_train
                labels = labels * (train_set.max_weight_train - train_set.min_weight_train) + train_set.min_weight_train

            train_loss += torch.sum(torch.abs(pred.view(-1) - labels.view(-1))).item()

        train_loss = train_loss / len(train_set)

        # run evaluation on validation set
        val_loss = 0.
        for it, data in enumerate(val_loader, 1):
            inputs, labels, idx = data
            inputs = inputs.to(cfg.MODEL.DEVICE)
            labels = labels.to(cfg.MODEL.DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    pred = model(inputs)

            # remap network output in range [0, 1] to weight in kg
            if cfg.INPUT.NORMALIZE_OUTPUT:
                pred = pred * (train_set.max_weight_train - train_set.min_weight_train) + train_set.min_weight_train
                labels = labels * (val_set.max_weight_train - val_set.min_weight_train) + val_set.min_weight_train

            val_loss += torch.sum(torch.abs(pred.view(-1) - labels.view(-1))).item()

        val_loss = val_loss / len(val_set)
        end_time = time.time()
        print('epoch', e, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_mae', '%0.3f' % train_loss,
              'val_mae', '%0.3f' % val_loss)

        validation_log[e, :] = [train_loss, val_loss]
        np.save(os.path.join(output_directory, "validation_history"), validation_log)
        torch.save(model.state_dict(), os.path.join(output_directory, 'model.pth'))


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
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    train(cfg, output_directory)
