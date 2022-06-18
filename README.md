# Varuna Hackathon

## Team: Noname

## Instructions

### Prepare environment

1. conda create -n varuna python=3.8; conda activate varuna
2. conda install -c conda-forge geopandas
3. pip install rasterio
4. conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
5. pip install tensorboard
6. pip install -U albumentations
7. pip install tqdm


### Use a Pretrained Model

xxxx


### Train your model

1. put the raw data into the ./data_raw/ folder
2. Run `python generate_mask.py` to generate raster masks from polygons
3. Run `python generate_data.py` to process the raw data into numpy arrays. The process involved date selection, band selections and random crop the big image into multiple smaller data samples.
4. Run `python train.py --name "my_model" --save` to train and save your model. 
5. Run `tensorbaord --logdir=runs` to visualize the training. 

### Code Structure

1. train.py : 
2. model.py
3. dataset.py
4. utils.py
5. generate_data.py
6. generate_mask.py
7. (TODO) test.py 
8. (TODO) test_prediction.csv 



