# Reproducing the submission on 18 June
# Result on test set: (Jaccard scores on each class) 0.30/0.26/0.36/0.22

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import cv2

# This model 

# 1. Download pre-trained model 
# from 

path_to_model = "models/model_1.pth.tar"

DEVICE = "cpu"
SIZE = (2051,2051)

# 2. self-contained model description

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        return self.conv(x)

class Standardize(nn.Module):
    def __init__(self, device="cuda"):
        super(Standardize, self).__init__()
        self.process = self.standardize_per_sample
        
    def forward(self, x):
        return self.process(x)

    def standardize_per_sample(self, x):
        N = x.shape[0]      
        x_view = x.reshape(N, -1)
        x_mean = torch.mean(x_view, dim=1).view(N,1,1,1) 
        x_std = 1e-5 + torch.std(x_view, dim=1).view(N,1,1,1)
        return (x - x_mean) / x_std

class NoNameUNET(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, 
        features=[64, 128, 256, 512]):

        super(NoNameUNET, self).__init__()
        self.preprocess = Standardize()
        self.out_channels = out_channels

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)     

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part 
        for feature in reversed(features):
            self.ups.append(
                    nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
                )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.preprocess(x)
   
        # unet model
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # the ConvTransposed
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)  #[b,c,h,w]
            x = self.ups[idx+1](concat_skip) # the DoubleConv

        return self.final_conv(x) 


# 3. download model 
def load_model(model_dir): 
    model = NoNameUNET(in_channels=12, out_channels=5).to(DEVICE)
    model.load_state_dict(torch.load(model_dir, map_location = torch.device(DEVICE))["state_dict"])
    model.eval()
    return model

model = load_model("models/model_1.pth.tar")


# 4. prepare input 
path_to_raw_data = "raw_data/sentinel-2-image/2021/20210106"
raster_path = "raw_data/sentinel-2-image/2021/20210106/IMG_DATA/47PQS_20210106_B02.jp2"

from generate_data import get_raw_data_paths

def combine_spectrum(paths, max_ndvi=None, max_evi=None):

    raw_spectrum = {}
    for band, path in paths.items():
        raw_spectrum[band] = cv2.resize(cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.uint16), dsize=SIZE)

    combined = np.dstack((raw_spectrum['aot'], raw_spectrum['b1'], raw_spectrum['b2'], 
                          raw_spectrum['b3'], raw_spectrum['b4'], raw_spectrum['b5'], 
                          raw_spectrum['b6'], raw_spectrum['b7'], raw_spectrum['b8'], 
                          raw_spectrum['b8a'], raw_spectrum['b11'], raw_spectrum['b12'],
                        ))

    return combined

paths = get_raw_data_paths('2021', '20210106')

combined = combine_spectrum(paths)
combined = combined.astype(np.float32)
combined = np.expand_dims(combined, axis=0)
x = np.transpose(combined, (0,3,1,2))

from utils import color_group
from PIL import Image

# 5. make a prediction on the whole image
with torch.no_grad():
    x = torch.tensor(x).to(DEVICE)
    pred = model(x).squeeze()

    pred_max = torch.argmax(pred, dim=0, keepdim=False) 
    pred_np = pred_max.cpu().numpy()

    pred_rgb = color_group[pred_np].astype(np.uint8)
    pred_rgb = Image.fromarray(pred_rgb)
    pred_rgb.save("v1_test_pred.png")



# 6. make CSV file
import rasterio
import geopandas as gpd
import pandas as pd
from test_v2 import poly_from_utm
from rasterio.features import rasterize

shape_path = "raw_data/testing_area/"

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
        max_value = values[sorted_index[-2]]

        class_predictions = np.vstack([class_predictions, 
                            np.array([num, max_value]).astype(int)])
        
    sorted_class_predictions = class_predictions[class_predictions[:, 0].argsort()]

    column_values = ['index', 'crop_type']
    df = pd.DataFrame(data = sorted_class_predictions, columns = column_values).astype(int)
    df.to_csv('reproduced_output_18_june.csv', index = False)
