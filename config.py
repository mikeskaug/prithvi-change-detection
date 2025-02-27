
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
    'out_dir': 'output',
    'gpu_devices': [0],
    'model_init_type': 'kaiming',
    'epochs': 10,
    'optimizer': 'adam',
    'LR': 0.011,
    'LR_policy': 'PolynomialLR',
    'LR_kwargs': {},
    'checkpoint_interval': 1,
    'resume': False,
    'resume_epoch': 1,
    'loss_weights': [ # should match the index order of DAMAGE_CLASS_IDS
        0.000711, # un-classified
        0.032151, # no-damage
        0.284842, # minor
        0.271633, # major
        0.410663, # destroyed
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
  
