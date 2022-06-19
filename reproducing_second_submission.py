# Reproducing the submission on 19 June 5pm

from test_v2 import *
import numpy as np
import torch
from model import *
from utils import *
from generate_data import *
from generate_mask import *
import geopandas as gpd
import pandas as pd


# 1. download model and processed data 
path_to_model = "models/model_2.pth.tar"
path_to_processed_ndvi_data = "data/max_ndvi.npy"
path_to_processed_evi_data  = "data/max_evi.npy"

model = load_model(path_to_model)
model.eval()


x = prepare_input('2021','20210220') 

pred_np = make_prediction(model, x)


testing_shape_path = "raw_data/testing_area/"
shape_path = testing_shape_path

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
    df.to_csv('reproduced_output_19_june.csv', index = False)