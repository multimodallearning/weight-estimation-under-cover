from yacs.config import CfgNode as CN

_C = CN()

_C.BASE_DIRECTORY = "../results"
_C.EXPERIMENT_NAME = ""

_C.INPUT = CN()
_C.INPUT.NUM_WORKERS = 4
_C.INPUT.NUM_POINTS = 0
_C.INPUT.NORMALIZE_OUTPUT = False
_C.INPUT.ROTATION_DEGREE = 0.
_C.INPUT.VOXELIZE = True
_C.INPUT.VOXEL_GRID_SHAPE = [48, 96, 32]
_C.INPUT.MIN_CLOUD_VALUES = [-0.86, -1.18, -0.46]
_C.INPUT.VOXEL_GRID_SIZE = [1.71, 2.36, 0.67]

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "CNN3D"
_C.MODEL.WEIGHT = ""

_C.MODEL.BPS = CN()
_C.MODEL.BPS.NUM_BASIS_POINTS = 2048
_C.MODEL.BPS.RADIUS = 1.2427
_C.MODEL.BPS.SEED = 13

_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.LR_MILESTONES = [60, 100]
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.WEIGHT_DECAY = 0.
_C.SOLVER.NUM_EPOCHS = 120
_C.SOLVER.BATCH_SIZE_TRAIN = 2
_C.SOLVER.BATCH_SIZE_TEST = 2
_C.SOLVER.USE_AMP = False

_C.SLP_DATASET = CN()
_C.SLP_DATASET.VAL_SPLIT = 'dana'
_C.SLP_DATASET.POSITION = 'supine'
_C.SLP_DATASET.COVER_CONDITION = 'uncover' #'uncover', 'cover1' or 'cover2'
_C.SLP_DATASET.USE_PATIENT_SEGMENTATION = True


def get_cfg_defaults():
  return _C.clone()