import os
import glob
import json
from collections import namedtuple

import click
import pandas as pd
from rasterio.features import rasterize
import rasterio
import numpy as np
from shapely import wkt

from dataset import load_raster
from config import DAMAGE_CLASS_IDS


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
            fill=DAMAGE_CLASS_IDS['un-classified'],
            dtype='uint8'
        )
    else:
        # no damage polygons in this sample
        mask_img = np.ones(mask_shape, dtype='uint8') * DAMAGE_CLASS_IDS['un-classified']

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
def target_masks(data_dir: str) -> None:
    '''
    Generate target image masks from the xBD target json files. Rasterizes the building polygons and damage labels
    to create one geotiff output for each sample with pixel values corresponding to DAMAGE_CLASS_ID
    '''
    labels_dir = os.path.join(data_dir, 'labels')
    for label_fl in [fl for fl in glob.glob(labels_dir + '/*post*.json')]:

        with open(label_fl) as fl:
            post_labels = json.load(fl)
            
        mask_img = mask_from_features(post_labels['features']['xy'], (1024, 1024))

        _, src_profile = load_raster(label_fl.replace('labels', 'images').replace('.json', '.tif'))
        write_mask(mask_img, src_profile, label_fl.replace('.json', '.tif'))
        

@click.command()
@click.option('--data_dir', help='Directory containing images and labels subdirectories')
@click.option('--out_file', help='Path of the output csv file', default='dataset_metadata.csv')
def metadata(data_dir: str, out_file: str) -> None:
    '''
    Create a csv metadata file with information about each input sample
    '''
    labels_dir = os.path.join(data_dir, 'labels')
    records = []
    for label_fl in [fl for fl in glob.glob(labels_dir + '/*post*.json')]:

        with open(label_fl) as fl:
            post_labels = json.load(fl)
        
        DamageShape = namedtuple('DamageShape', ['shape', 'type'])
        shapes = [DamageShape(wkt.loads(feature['wkt']), feature['properties']['subtype']) for feature in post_labels['features']['xy']]

        with open(label_fl.replace('post', 'pre')) as fl:
            pre_labels = json.load(fl)

        _, src_profile = load_raster(label_fl.replace('labels', 'images').replace('.json', '.tif'))
        image_origin = rasterio.transform.AffineTransformer(src_profile['transform']).xy(0,0)

        sample_metadata ={
            'pre_date': pre_labels['metadata']['capture_date'],
            'post_date': post_labels['metadata']['capture_date'],
            'num_buildings': len(pre_labels['features']['xy']),
            'image_name': pre_labels['metadata']['img_name'].replace('.png', '.tif'),
            'image_lon': image_origin[0],
            'image_lat': image_origin[1],
            'disaster': pre_labels['metadata']['disaster'],
            'disaster_type': pre_labels['metadata']['disaster_type'],
            'xy_area_no-damage': sum([shape.shape.area for shape in shapes if shape.type == 'no-damage']),
            'xy_area_minor-damage': sum([shape.shape.area for shape in shapes if shape.type == 'minor-damage']),
            'xy_area_major-damage': sum([shape.shape.area for shape in shapes if shape.type == 'major-damage']),
            'xy_area_destroyed': sum([shape.shape.area for shape in shapes if shape.type == 'destroyed']),
            'xy_area_un-classified': sum([shape.shape.area for shape in shapes if shape.type == 'un-classified']),
            'pre_gsd': pre_labels['metadata']['gsd'],
            'pre_sun_azimuth': pre_labels['metadata']['sun_azimuth'],
            'pre_sun_elevation': pre_labels['metadata']['sun_elevation'],
            'pre_off_nadir_angle': pre_labels['metadata']['off_nadir_angle'],
            'post_gsd': post_labels['metadata']['gsd'],
            'post_sun_azimuth': post_labels['metadata']['sun_azimuth'],
            'post_sun_elevation': post_labels['metadata']['sun_elevation'],
            'post_off_nadir_angle': post_labels['metadata']['off_nadir_angle'],
        }
        records.append(sample_metadata)

    out_df = pd.DataFrame(records)
    out_df.to_csv(os.path.join(data_dir, out_file), index=False)


cli.add_command(target_masks)
cli.add_command(metadata)

if __name__ == '__main__':
    cli()