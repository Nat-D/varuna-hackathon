import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import *
from utils import *
from dataset import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='', help='Experiment name')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--bs', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=10000, help='Number of training epoch')
parser.add_argument('--preprocess', type=str, default='PS', help='How to standardize input')
parser.add_argument('--save', action='store_true', default=False, help='save model')
parser.add_argument('--load', action='store_true', default=False, help='load model')
parser.add_argument('--model', type=str, default='UNET', help='Select your model')
args = parser.parse_args()

# Experiment log
LOG_DIR = f"runs/{args.name}"

# Hyper parameters
LEARNING_RATE = args.lr
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.bs
BATCH_SIZE_VAL = 4
NUM_EPOCHS = args.epochs
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
VAL_HEIGHT = 512
VAL_WIDTH = 512
MIN_MAX_HEIGHT = (224,256)
PIN_MEMORY = True


TRAIN_IMG_DIR = "data/train/img/"
TRAIN_MASK_DIR = "data/train/mask/"
VAL_IMG_DIR = "data/val/img/"
VAL_MASK_DIR = "data/val/mask" 

def train_fn(loader, model, optimizer, loss_fn, scaler, logger):

    # decorate loader with tqdm
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        # forward (float16 )
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets) 

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()   

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # log training 
        logger.log_step(loss.item())

def main():
    train_transform = A.Compose([
            A.ToFloat(max_value=65535.0), # support uint16
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.RandomSizedCrop(min_max_height=MIN_MAX_HEIGHT, 
                              height=IMAGE_HEIGHT, width=IMAGE_WIDTH, p=0.2),
            A.Rotate(limit= 90, p=1.0),  
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(p=0.8, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), # try grid dropout, randomgridshuffle
            ToTensorV2()
        ])

    val_transforms = A.Compose([
            A.ToFloat(max_value=65535.0),
            A.Resize(height=VAL_HEIGHT, width=VAL_WIDTH), #hack
            ToTensorV2()
        ])


    preprocess = Standardize(args.preprocess, 
                             device=DEVICE)

    if args.model == 'UNET':
        model = NoNameUNET(in_channels=12, out_channels=5, preprocess=preprocess).to(DEVICE)
    else:
        raise NotImplementedError("No model")

    # mask out the unknown class
    weight = torch.tensor([0, 1, 1, 1, 1]).float().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        BATCH_SIZE_VAL,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
        )

    if args.load:
        load_checkpoint(torch.load(f'{LOG_DIR}/my_checkpoint.pth.tar'), model)

    logger = Logger(device=DEVICE, log_dir=LOG_DIR)
    

    for epoch in range(NUM_EPOCHS): 
        train_fn(train_loader, model, optimizer, loss_fn, scaler, logger)
        logger.log_epoch(val_loader, model, optimizer)

        # save every 10 epoch
        if epoch % 10 == 0:
            if args.save:
                check_point = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(check_point, filename=f'{LOG_DIR}/my_checkpoint.pth.tar')
        
        if epoch == 100:
            check_point = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(check_point, filename=f'{LOG_DIR}/model_at_100epoch.pth.tar')

if __name__ == "__main__":
    main()