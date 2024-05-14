
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
    'decoder_depth': 8,
    'decoder_embed_dim': 512,
    'decoder_num_heads': 16,
    'depth': 12,
    'embed_dim': 768,
    'img_size': 1024,
    'in_chans': 3,
    'num_frames': 2,
    'num_heads': 12,
    'patch_size': 16,
    'tubelet_size': 1,
    'num_classes': 5 # No damage, minor, major, destroyed, unknown
}
  