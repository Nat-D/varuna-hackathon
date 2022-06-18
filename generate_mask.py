"""
Convert Polygons into Raster Masks
"""

import os 
import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import geopandas as gpd

from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import color_group

## tutorial from https://lpsmlgeo.github.io/2019-09-22-binary_mask/


raster_path = "raw_data/sentinel-2-image/2021/20210106/IMG_DATA/47PQS_20210106_B02.jp2"

def generate_mask(shape_path, src, test=False):
    raster_img = src.read()
    raster_meta = src.meta
    print(raster_meta)

    # load shape
    train_df = gpd.read_file(shape_path)

    # verify crs coordinate system
    print(f"CRS vector: {train_df.crs}, CRS raster: {src.crs}")

    # generate mask
    def poly_from_utm(polygon, transform):
        poly_pts = []
        poly = cascaded_union(polygon)

        for i in np.array(poly.exterior.coords):

            # convert polygons to the image crs
            poly_pts.append(~transform * tuple(i))

        # generate a polygon object
        new_poly = Polygon(poly_pts)

        return new_poly
    if not test:
        poly_shp_train = {}
        poly_shp_train['1'] = []
        poly_shp_train['2'] = []
        poly_shp_train['3'] = []
        poly_shp_train['4'] = []

        poly_shp_val = {}
        poly_shp_val['1'] = []
        poly_shp_val['2'] = []
        poly_shp_val['3'] = []
        poly_shp_val['4'] = []

        im_size = (src.meta['height'], src.meta['width'])
        
        count = 0   
        count_val = 0
        for num, row in train_df.iterrows():
            rand_train = np.random.choice([True, False], p=[0.8, 0.2])
            for crop_type in ['1','2','3','4']:
                if row['crop_type'] == crop_type:
                    poly = poly_from_utm(row['geometry'], src.meta['transform'])
                    
                    if rand_train:
                        poly_shp_train[crop_type].append(poly)
                        count += 1
                    else:
                        poly_shp_val[crop_type].append(poly)
                        count_val += 1

        print(count)
        print(count_val)
        return poly_shp_train, poly_shp_val
    else: 
        print('test version')
        poly_shp_test = []
        im_size = (src.meta['height'], src.meta['width'])

        for num, row in train_df.iterrows():
            poly = poly_from_utm(row['geometry'], src.meta['transform'])                    
            poly_shp_test.append(poly)
 
        return poly_shp_test


def package(poly_shp, name, src):
    mask = {}
    mask['1'] = []
    mask['2'] = []
    mask['3'] = []
    mask['4'] = []
    im_size = (src.meta['height'], src.meta['width'])

    for crop_type in ['1','2','3','4']:
        mask[crop_type] = rasterize(shapes=poly_shp[crop_type], out_shape=im_size)

    
    color_mask = np.zeros([src.meta['height'], src.meta['width']])

    # todo: optimize this maybe?
    for i in range(src.meta['height']):
        for j in range(src.meta['width']):
            if (mask['1'][i,j] +  mask['2'][i,j]
             + mask['3'][i,j] + mask['4'][i,j]) <= 1:
                color_mask[i,j] = 1 *mask['1'][i,j] + \
                                  2 *mask['2'][i,j] + \
                                  3 *mask['3'][i,j] + \
                                  4 *mask['4'][i,j] 
            
    color_mask = color_mask.astype(np.uint8)
    np.save(name+'label.npy', color_mask)
    cv2.imwrite(name+'mask_color.png', color_group[color_mask])

        
if __name__ == "__main__":
    
    training_shape_path = "raw_data/training_area/"
    testing_shape_path = "raw_data/testing_area/"

    with rasterio.open(raster_path, "r") as src:
        poly_shp_train, poly_shp_val = generate_mask(training_shape_path, src)

        package(poly_shp_train, 'raw_data/train_', src)
        package(poly_shp_val, 'raw_data/val_', src)

        poly_shp_test = generate_mask(testing_shape_path, src, test=True)
        im_size = (src.meta['height'], src.meta['width'])
        test_mask = rasterize(shapes=poly_shp_test, out_shape=im_size)
        cv2.imwrite('test_mask.png', test_mask*255)
        np.save('test_label.npy', test_mask)

        print('Mask Generated -- see /raw_data/')
        