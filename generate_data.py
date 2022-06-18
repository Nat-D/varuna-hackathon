"""
Generate dataset
"""

import numpy as np
import os 
import cv2 
import numpy

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

def combine_spectrum(paths):

    raw_spectrum = {}
    for band, path in paths.items():
        raw_spectrum[band] = cv2.resize(cv2.imread(path, cv2.IMREAD_ANYDEPTH), dsize=SIZE)
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



def random_crop_data(crop_size, img, label):
    rand_h = int( (SIZE[0]-crop_size) * np.random.rand() ) 
    rand_w = int( (SIZE[1]-crop_size) * np.random.rand() )# uniform random
    
    croped_img = img[rand_h:rand_h+crop_size, rand_w:rand_w+crop_size, :]
    croped_label = label[rand_h:rand_h+crop_size, rand_w:rand_w+crop_size]

    return croped_img, croped_label


if __name__ == "__main__":

    paths = get_raw_data_paths('2021', '20210106')
    combined = combine_spectrum(paths)

    print(combined.shape) #(2051, 2051, 12)
    
    train_label = np.load("raw_data/v2_train_label.npy") # generated from generate_mask.py
    val_label = np.load("raw_data/v2_val_label.npy")


    # selected without cloud
    days = [20210101
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
            ,20210416
            ,20210705]
    day_idx = 0
    for day in days:
        paths = get_raw_data_paths('2021', str(day))
        combined = combine_spectrum(paths)

        num_crop_per_img = 40
        for i in range(num_crop_per_img): # as a test let's do 20
            img, label = random_crop_data(crop_size=512, img=combined, label=train_label)
            np.save(f'data/train/img/{i+num_crop_per_img*day_idx}.npy', img)
            np.save(f'data/train/mask/{i+num_crop_per_img*day_idx}_label.npy', label)
        
        num_crop_per_img_val = 5
        for i in range(5):
            img, label = random_crop_data(crop_size=512, img=combined, label=val_label)
            np.save(f'data/val/img/{i+num_crop_per_img_val*day_idx}.npy', img)
            np.save(f'data/val/mask/{i+num_crop_per_img_val*day_idx}_label.npy', label)
        
        day_idx += 1


