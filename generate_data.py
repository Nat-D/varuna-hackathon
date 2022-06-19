"""
Generate dataset
"""

import numpy as np
import os 
import cv2 
import numpy
import features

from train import IMAGE_HEIGHT, VAL_HEIGHT 

SIZE = (2051,2051)

raw_data_dir = "raw_data/sentinel-2-image/"

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

def combine_spectrum(paths, max_ndvi=None, max_evi=None):

    raw_spectrum = {}
    for band, path in paths.items():
        # prone to bug/ check whether cv2 read int16 correctly.
        raw_spectrum[band] = cv2.resize(cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.uint16), dsize=SIZE)
    # ignore scl/tci/wvp just to keep it raw
    
    # add max_ndvi, max_evi to incorporate temporal information    
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
                          raw_spectrum['b12'],
                          max_ndvi,
                          max_evi
                        ))

    return combined

def get_raw_spectrum(paths):
    raw_spectrum = {}
    for band, path in paths.items():
        raw_spectrum[band] = cv2.resize(cv2.imread(path, cv2.IMREAD_ANYDEPTH), dsize=SIZE).astype(np.float16)   
    return raw_spectrum



def max_ndvi_across_time():

    img_ndvi = np.zeros(SIZE)

    for year in ['2021']: #['2020', '2021']:
        dates = os.listdir(os.path.join(raw_data_dir, year))
        for date in dates:

            paths = get_raw_data_paths(year, date)
            try:
                raw_spectrum = get_raw_spectrum( {'b4': paths['b4'], 'b8': paths['b8']})
                current_img_ndvi = ndvi(raw_spectrum)
                img_ndvi = np.maximum(img_ndvi, current_img_ndvi)
            except:
                pass

    return img_ndvi

def max_evi_across_time():

    img_evi = np.zeros(SIZE)

    for year in ['2021']: #['2020', '2021']:
        dates = os.listdir(os.path.join(raw_data_dir, year))
        for date in dates:

            paths = get_raw_data_paths(year, date)
            try:
                raw_spectrum = get_raw_spectrum( {'b2' :paths['b2'],'b4': paths['b4'], 'b8': paths['b8']})
                current_img_evi = evi(raw_spectrum)
                img_evi = np.maximum(img_evi, current_img_evi)
            except:
                pass

    return img_evi



def random_crop_data(crop_size, img, label):
    rand_h = int( (SIZE[0]-crop_size) * np.random.rand() ) 
    rand_w = int( (SIZE[1]-crop_size) * np.random.rand() )# uniform random
    
    croped_img = img[rand_h:rand_h+crop_size, rand_w:rand_w+crop_size, :]
    croped_label = label[rand_h:rand_h+crop_size, rand_w:rand_w+crop_size]

    return croped_img, croped_label


def create_dataset():
    from pathlib import Path
    Path("data/train/img/").mkdir(parents=True, exist_ok=True)
    Path("data/train/mask/").mkdir(parents=True, exist_ok=True)
    Path("data/val/img/").mkdir(parents=True, exist_ok=True)
    Path("data/val/mask/").mkdir(parents=True, exist_ok=True)


    train_label = np.load("raw_data/train_label.npy") # generated from generate_mask.py
    val_label = np.load("raw_data/val_label.npy")


    # Hand selected for good quality images without clouds
    days = [ 20210101
            ,20210106
            ,20210111
            ,20210116
            ,20210126
            ,20210205
            ,20210215
            ,20210220
            ,20210307
            ,20210312
            ,20210327
            ,20210416]
    day_idx = 0

    train_label = np.load("raw_data/train_label.npy") # generated from generate_mask.py
    val_label = np.load("raw_data/val_label.npy")


    max_ndvi = max_ndvi_across_time()
    max_evi = max_evi_across_time()

    print('generating dataset')

    for day in days:
        paths = get_raw_data_paths('2021', str(day))
        combined = combine_spectrum(paths, max_ndvi, max_evi) 

        # for some reason some samples are missing
        num_crop_per_img = 60
        for i in range(num_crop_per_img): # as a test let's do 20
            img, label = random_crop_data(crop_size=IMAGE_HEIGHT, img=combined, label=train_label)
            np.save(f'data/train/img/{i+num_crop_per_img*day_idx}.npy', img)
            np.save(f'data/train/mask/{i+num_crop_per_img*day_idx}_label.npy', label)
        
        num_crop_per_img_val = 8 #20
        for i in range(num_crop_per_img_val):
            img, label = random_crop_data(crop_size=VAL_HEIGHT, img=combined, label=val_label)
            np.save(f'data/val/img/{i+num_crop_per_img_val*day_idx}.npy', img)
            np.save(f'data/val/mask/{i+num_crop_per_img_val*day_idx}_label.npy', label)
        
        day_idx += 1


    print('Succesfully create a dataset')

if __name__ == "__main__":

    create_dataset()