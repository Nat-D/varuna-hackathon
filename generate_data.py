"""
Generate dataset
"""

import numpy as np
import os 
import cv2 
import numpy

raw_data_dir = "raw_data/sentinel-2-image/"
train_label = np.load("raw_data/train_label.npy") # generated from generate_mask.py
val_label = np.load("raw_data/val_label.npy")

def get_raw_data_paths(year, date):
    paths = {}
    paths['aot'] = os.path.join(raw_data_dir, year, date, 'IMG_DATA/47PQS_'+ date + '_AOT.jp2')
    paths['b1'] = os.path.join(raw_data_dir,  year, date, 'IMG_DATA/47PQS_'+ date + '_B01.jp2')
    paths['b2'] = os.path.join(raw_data_dir,  year, date, 'IMG_DATA/47PQS_'+ date + '_B02.jp2')
    paths['b3'] = os.path.join(raw_data_dir,  year, date, 'IMG_DATA/47PQS_'+ date + '_B03.jp2')
    paths['b4'] = os.path.join(raw_data_dir,  year, date, 'IMG_DATA/47PQS_'+ date + '_B04.jp2')
    paths['b5'] = os.path.join(raw_data_dir,  year, date, 'IMG_DATA/47PQS_'+ date + '_B05.jp2')
    paths['b6'] = os.path.join(raw_data_dir,  year, date, 'IMG_DATA/47PQS_'+ date + '_B06.jp2')
    paths['b7'] = os.path.join(raw_data_dir,  year, date, 'IMG_DATA/47PQS_'+ date + '_B07.jp2')
    paths['b8'] = os.path.join(raw_data_dir,  year, date, 'IMG_DATA/47PQS_'+ date + '_B08.jp2')
    paths['b8a'] = os.path.join(raw_data_dir, year, date, 'IMG_DATA/47PQS_'+ date + '_B8A.jp2')
    paths['b11'] = os.path.join(raw_data_dir, year, date, 'IMG_DATA/47PQS_'+ date + '_B11.jp2')
    paths['b12'] = os.path.join(raw_data_dir, year, date, 'IMG_DATA/47PQS_'+ date + '_B12.jp2')
    paths['scl'] = os.path.join(raw_data_dir, year, date, 'IMG_DATA/47PQS_'+ date + '_SCL.jp2')
    paths['tci'] = os.path.join(raw_data_dir, year, date, 'IMG_DATA/47PQS_'+ date + '_TCI.jp2')
    paths['wvp'] = os.path.join(raw_data_dir, year, date, 'IMG_DATA/47PQS_'+ date + '_WVP.jp2')
    return paths

def combine_spectrum(paths):

    raw_spectrum = {}
    for band, path in paths.items():
        raw_spectrum[band] = cv2.resize(cv2.imread(path, cv2.IMREAD_ANYDEPTH), dsize=(2051,2051))
    # ignore scl/tci/wvp just to keep it raw
    combined = np.dstack((raw_spectrum['aot'], 
                          raw_spectrum['b1'], 
                          raw_spectrum['b2'], 
                          raw_spectrum['b3'], 
                          raw_spectrum['b4'], 
                          raw_spectrum['b5'], 
                          raw_spectrum['b6'], 
                          raw_spectrum['b7'], 
                          raw_spectrum['b8'], 
                          raw_spectrum['b8a'], 
                          raw_spectrum['b11'], 
                          raw_spectrum['b12']))
    return combined


paths = get_raw_data_paths('2021', '20210106')
combined = combine_spectrum(paths)

print(combined.shape)