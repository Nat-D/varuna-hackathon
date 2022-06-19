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



### Train your model

1. put the raw data into the ./data_raw/ folder
2. Run `python generate_mask.py` to generate raster masks from polygons
3. Run `python generate_data.py` to process the raw data into numpy arrays. The process involved date selection, band selections and random crop the big image into multiple smaller data samples.
4. Run `python train.py --name "my_model" --save` to train and save your model. 
5. Run `tensorbaord --logdir=runs` to visualize the training. 



### Use pretrained model to predict the test shape

1. Prepare your data into appropriate folders
2. Run `python test_v2.py`


### Reproducing the CSV results

1. Our pretrained model can be downloaded here:
    https://drive.google.com/drive/folders/1drKdEeyS_zlUwh0ncXA4hyCx7ZHXhVx7?usp=sharing

2. Run `python reproducing_first_submission.py` to reproduce the first submission
<img src="v1_test_pred.png " width="200">

3. Run `python reproducing_second_submission.py` to reproduce the 2nd submission
<img src="v2_test_pred.png " width="200">
