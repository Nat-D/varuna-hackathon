"""
Run this file to perform inference on test areas.
output: test_output.csv
"""

import numpy as np
import torch
from model import *
from utils import *
from generate_data import *
from generate_mask import *
import geopandas as gpd
import pandas as pd


DEVICE = "cpu"


def load_model(model_dir):
    preprocess = Standardize('PS', device= DEVICE) 
    model = NoNameUNET(in_channels=14, out_channels=5, preprocess=preprocess).to(DEVICE)
    model.load_state_dict(torch.load(model_dir, map_location = torch.device(DEVICE))["state_dict"])
    model.eval()

    return model 

def load_max_ndvi_evi():
    max_ndvi = np.load('data/max_ndvi.npy')
    max_evi  = np.load('data/max_evi.npy')
    return max_ndvi, max_evi

def save_max_ndvi_evi():
    max_ndvi = max_ndvi_across_time()
    max_evi = max_evi_across_time()
    np.save('data/max_ndvi.npy', max_ndvi)
    np.save('data/max_evi.npy', max_evi)


def prepare_input(year, date):
    paths = get_raw_data_paths(year, date)
    max_ndvi, max_evi = load_max_ndvi_evi()
    combined = combine_spectrum(paths, max_ndvi, max_evi)
    combined = combined.astype(np.float32)
    combined = np.expand_dims(combined, axis=0)
    combined = np.transpose(combined, (0,3,1,2))
    return combined


def make_prediction(model, x):

    with torch.no_grad():
            x = torch.tensor(x).to(DEVICE)
            pred = model(x).squeeze()

    pred_max = torch.argmax(pred, dim=0, keepdim=False) 
    pred_np = pred_max.cpu().numpy()

    pred_rgb = color_group[pred_np].astype(np.uint8)
    pred_rgb = Image.fromarray(pred_rgb)
    pred_rgb.save("test_pred.png")

    return pred_np


def poly_from_utm(polygon, transform):
        poly_pts = []
        poly = cascaded_union(polygon)

        for i in np.array(poly.exterior.coords):

            # convert polygons to the image crs
            poly_pts.append(~transform * tuple(i))

        # generate a polygon object
        new_poly = Polygon(poly_pts)

        return new_poly


if __name__ == "__main__":

    #save_max_ndvi_evi() # only do this once

    model = load_model("runs/finalfinal/my_checkpoint-2000.pth.tar")
    model.eval()

    x = prepare_input('2021','20210416') # todo: select a better date 

    pred_np = make_prediction(model, x)
    

    training_shape_path = "raw_data/training_area/"
    shape_path = training_shape_path

    #testing_shape_path = "raw_data/testing_area/"
    #shape_path = testing_shape_path
    
    class_predictions = np.zeros((0, 2))

    with rasterio.open(raster_path, "r") as src:
        im_size = (src.meta['height'], src.meta['width'])    

        train_df = gpd.read_file(shape_path)
        for num, row in train_df.iterrows():
            
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            test_mask = rasterize(shapes=[poly], out_shape=im_size)
            
            masked_pred = pred_np * test_mask

            values, counts  = np.unique(masked_pred, return_counts=True)
            sorted_index = sorted(range(len(counts)), key=lambda k: counts[k])
            try:
                max_value = values[sorted_index[-2]]
            except:
                max_value = 1 # class unknown/ hack! use 1 for now
            
            class_predictions = np.vstack([class_predictions, 
                                np.array([num, max_value]).astype(int)])
            
        sorted_class_predictions = class_predictions[class_predictions[:, 0].argsort()]

        column_values = ['index', 'crop_type']
        df = pd.DataFrame(data = sorted_class_predictions, columns = column_values).astype(int)
        #df.to_csv('test_output_19_june.csv', index = False)
        df.to_csv('train_prediction.csv', index = False)

