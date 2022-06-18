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

## tutorial from https://lpsmlgeo.github.io/2019-09-22-binary_mask/

raster_path = "C:/Users/NataphopT/Desktop/ARVHackathon/VarunaHackathon/raw-data/sentinel-2-image/2021/20210106/IMG_DATA/47PQS_20210106_B02.jp2"

def generate_test_mask(shape_path, src, test=False):
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

    print('test version')
    poly_shp_test = {}
    im_size = (src.meta['height'], src.meta['width'])

    for num, row in train_df.iterrows():
        temp =[]
        poly = poly_from_utm(row['geometry'], src.meta['transform'])             
        poly_shp_test[num] = poly
        temp.append(poly)
        test_mask = rasterize(shapes=temp, out_shape=im_size)
        cv2.imwrite(f'./test_output/img/{num}.png', test_mask*255)
        np.save(f'./test_output/mask/{num}.npy', test_mask)

    return poly_shp_test


        
if __name__ == "__main__":
    
    testing_shape_path = "C:/Users/NataphopT/Desktop/ARVHackathon/VarunaHackathon/raw-data/testing_area/"

    with rasterio.open(raster_path, "r") as src:
        poly_shp_test = generate_test_mask(testing_shape_path, src, test=True)

