import sys
sys.path.append('../')
import argparse
import os
import torch
from torch.utils.data import DataLoader
from models.utils import create_model
from data.slp_weight_dataset import SLPWeightDataset
from configs.defaults import get_cfg_defaults


def test(cfg, model_path):
    # DATA
    train_set = SLPWeightDataset(cfg, phase='train')
    val_set = SLPWeightDataset(cfg, phase='val')
    val_loader = DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE_TEST, num_workers=cfg.INPUT.NUM_WORKERS,
                            shuffle=False, drop_last=False)

    # MODEL
    model = create_model(cfg).to(cfg.MODEL.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    val_loss = 0.
    for it, data in enumerate(val_loader, 1):
        inputs, labels, idx = data
        inputs = inputs.to(cfg.MODEL.DEVICE)
        labels = labels.to(cfg.MODEL.DEVICE)

        with torch.no_grad():
            pred = model(inputs)

        if cfg.INPUT.NORMALIZE_OUTPUT:
            pred = pred * (train_set.max_weight_train - train_set.min_weight_train) + train_set.min_weight_train
            labels = labels * (val_set.max_weight_train - val_set.min_weight_train) + val_set.min_weight_train
        val_loss += torch.sum(torch.abs(pred.view(-1) - labels.view(-1))).item()

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
        "--model-path",
        default="",
        metavar="FILE",
        help="path to pre-trained model",
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

    if args.model_path == "":
        working_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
        model_path = os.path.join(working_directory, 'model.pth')
    else:
        model_path = args.model_path
    if not os.path.isfile(model_path):
        raise ValueError('There is no pre-trained model at the specified path. Set the model path correctly or'
                         ' run training first.')

    test(cfg, model_path)