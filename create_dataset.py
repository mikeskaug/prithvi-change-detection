import os
import glob
import json

import click
from rasterio.features import rasterize
import rasterio
import numpy as np
from shapely import wkt

from dataset import load_raster

DAMAGE_CLASS_IDS ={
    'no-damage': 0,
    'un-classified': 0, # I'm not sure how to properly label this low frequency class.
    'minor-damage': 1,
    'major-damage': 2,
    'destroyed': 3
}


def mask_from_features(features: list, mask_shape: tuple) -> np.array:
    '''
    :param features: The 'xy' geojson feature list from the labels json
    :param size: A tuple of the (width, height) size of the output array
    :returns: A numpy array with the polygons filled with correspoding damage class id 
    '''
    
    shapes = [(wkt.loads(feature['wkt']), DAMAGE_CLASS_IDS[feature['properties']['subtype']]) for feature in features]
    
    if len(shapes) > 0:
        mask_img = rasterize(
            shapes,
            out_shape=mask_shape,
            fill=DAMAGE_CLASS_IDS['no-damage'],
            dtype='uint8'
        )
    else:
        # no damage polygons in this sample
        mask_img = np.zeros(mask_shape, dtype='uint8')

    return mask_img


def write_mask(mask_img: np.array, src_profile: dict, out_file: str) -> None:
    '''
    Write a single band geotiff with the damage mask with pixel values corresponding to the DAMMAGE_CLASS_ID

    :param mask_img: numpy array with the mask
    :param src_profile: the rasterio src.profile from the post image
    :param out_file: path to write the geotiff output 
    '''
    src_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=99
    )

    with rasterio.open(out_file, 'w', **src_profile) as dst:
        dst.write(mask_img.astype(rasterio.uint8), 1)


@click.group()
def cli():
    pass


@click.command()
@click.option('--data_dir', help='Directory containing images and labels subdirectories to process')
def target_masks(data_dir):
    '''
    Generate target image masks from the xBD target json files. Rasterizes the building polygons and damage labels
    to create one geotiff output for each sample with pixel values corresponding to DAMAGE_CLASS_ID
    '''
    labels_dir = os.path.join(data_dir, 'labels')
    for label_fl in [fl for fl in glob.glob(labels_dir + '/*post*.json')]:
        print(label_fl)
        with open(label_fl) as fl:
            post_labels = json.load(fl)
            
            mask_img = mask_from_features(post_labels['features']['xy'], (1024, 1024))

            _, src_profile = load_raster(label_fl.replace('labels', 'images').replace('.json', '.tif'))
            write_mask(mask_img, src_profile, label_fl.replace('.json', '.tif'))
        


cli.add_command(target_masks)

if __name__ == '__main__':
    cli()