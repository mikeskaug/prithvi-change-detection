
import numpy as np


DATA_MEAN = np.array([
    81.70725048,
    91.98094707,
    60.39953364
]).reshape((3, 1, 1))

DATA_STD = np.array([
    34.17098048,
    31.98962165,
    34.70514339
]).reshape((3, 1, 1))

MODEL_ARGS = {
    'embed_dim': 768,
    'img_size': 1024,
    'input_channels': 3,
    'num_frames': 2,
    'patch_size': 16,
    'tubelet_size': 1,
    'num_classes': 5, # No damage, minor, major, destroyed, un-classified
    'working_dir': 'output',
    'out_dir': 'output',
    'gpu_devices': [0],
    'model_init_type': 'kaiming',
    'epochs': 1,
    'optimizer': 'adam',
    'LR': 0.011,
    'LR_policy': 'PolynomialLR',
    'checkpoint_interval': 1,
    'resume': False,
    'resume_epoch': None,
    'loss_weights': [
        0.0016, # un-classified
        0.0359, # no-damage
        0.2604, # minor
        0.2448, # major
        0.4572, # destroyed
    ]
}

DAMAGE_CLASS_IDS = {
    'un-classified': 0,
    'no-damage': 1,
    'minor-damage': 2,
    'major-damage': 3,
    'destroyed': 4
}

CLASS_MAPPING = {val: key for key, val in DAMAGE_CLASS_IDS.items()}
  