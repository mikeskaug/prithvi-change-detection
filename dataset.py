import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from config import DATA_MEAN, DATA_STD


class XBDDataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            transform=None, 
            target_transform=None
        ):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(data_dir, 'images', fl) for fl in os.listdir(os.path.join(data_dir, 'images')) if 'pre' in fl]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pre_image_path = self.image_paths[idx]
        pre_image, _ = load_raster(pre_image_path)
        pre_image = preprocess_image(pre_image)

        post_image_path = pre_image_path.replace('pre', 'post')
        post_image, _ = load_raster(post_image_path)
        post_image = preprocess_image(post_image)

        both_frames = np.stack([pre_image, post_image], axis=1)
        
        input = torch.from_numpy(both_frames).to(torch.float32)

        labels, _ = load_raster(post_image_path.replace('images', 'labels'))
        
        return input , labels


def preprocess_image(image: np.array):
    # normalize image
    normalized = ((image - DATA_MEAN) / DATA_STD)

    # reverse band order RGB -> BGR to match HLS data B02, B03, B04
    bgr = np.flip(normalized, axis=0)

        
    return bgr


def load_raster(path, out_size=None):
    with rasterio.open(path) as src:
        if out_size:
            out_shape = (
                src.count,
                out_size,
                out_size
            )
        else:
            out_shape = None
            
        img = src.read(
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.bilinear
        )

        src_profile = src.profile

    return img, src_profile