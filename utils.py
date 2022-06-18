import torch 
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from dataset import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


color_group = np.array([[0,0,0],  # black - unknown
                       [255,0,0], # red - cassava 
                       [0, 255, 0],  # green - rice
                       [0, 0, 255],  # blue - maize 
                       [255, 255, 255],  # white - sugarcane 
                       [100, 100, 100]   # grey 
                       ]) 


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    batch_size_val,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    ):

    train_ds = VarunaData(
            image_dir=train_dir,
            mask_dir=train_maskdir,
            transform=train_transform,
        )
    train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True
        )

    val_ds = VarunaData(
            image_dir=val_dir,
            mask_dir=val_maskdir,
            transform=val_transform,
        )

    val_loader = DataLoader(
            val_ds,
            batch_size=batch_size_val,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False
        )

    return train_loader, val_loader


class Logger():
    def __init__(self, device="cuda", log_dir='runs'):
        
        self.writer = SummaryWriter(log_dir)

        self.device = device
        self.accumulate_training_loss = 0.0
        self.training_step = 0
        self.epoch_num_step = 0
        # compute validation loss only on crops type, ignore background and unknown
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([0,1,1,1,1,0]).float().to(device))

    def compute_precision(self, true_pos, false_pos, false_neg):
        return true_pos / (true_pos + false_pos + 1e-5)

    # we probably want high recall
    # how many pixel of plant we correctly retrieved
    def compute_recall(self, true_pos, false_pos, false_neg):
        return true_pos / (false_neg + true_pos + 1e-5)

    # iou becomes somewhat useless in the present of unknown class
    # false_positive becomes unreliable because it might be true_positive
    def compute_iou(self, true_pos, false_pos, false_neg):
        return true_pos / (true_pos + false_pos + false_neg + 1e-5)


    def validation(self, loader, model):
        model.eval()

        total_loss = 0
        num_step = 0
        #total_iou_for_each_class = 0
        total_recall_for_each_class = 0

        with torch.no_grad():
            for x,y in loader:
                x = x.to(self.device)
                y = y.long().to(self.device)

                preds = model(x)
                loss = self.loss_fn(preds, y)
    
                total_loss += loss
                num_step += 1

                preds_max = torch.argmax(preds, dim=1, keepdim=False) #[B, H, W]
                preds_max = F.one_hot(preds_max, num_classes=model.out_channels) #[B, H, W, C]

                y = F.one_hot(y, num_classes=model.out_channels) #[B, H, W, C]
                
                ones = torch.ones_like(preds_max)
                zeros = torch.zeros_like(preds_max)

                true_pos = torch.logical_and((preds_max == ones), (preds_max == y)) 
                true_pos = torch.sum(true_pos, dim=(1,2)) # [B, C]

                false_pos = torch.logical_and((preds_max == ones),(preds_max != y))
                false_pos = torch.sum(false_pos, dim=(1,2))

                false_neg = torch.logical_and((preds_max == zeros), (preds_max != y))
                false_neg = torch.sum(false_neg, dim=(1,2))
                
                #iou = self.compute_iou(true_pos, false_pos, false_neg) 
                #total_iou_for_each_class += torch.sum(iou, dim=0)
                
                recall = self.compute_recall(true_pos, false_pos, false_neg)
                total_recall_for_each_class += torch.sum(recall, dim=0)

        self.writer.add_scalar("Loss/Average_validation_loss", 
                                total_loss/num_step, 
                                self.training_step)

        #for cls in range(model.out_channels):
        #    self.writer.add_scalar(f"IoU/Average_iou_class_{cls}",
        #                           total_iou_for_each_class[cls]/(num_step * loader.batch_size),
        #                           self.training_step)
        #self.writer.add_scalar("meanIoU", 
        #    torch.sum(total_iou_for_each_class[1:])/(num_step * loader.batch_size * (model.out_channels-1)),
        #    self.training_step)

        for cls in range(model.out_channels):
            self.writer.add_scalar(f"Recall/Average_recall_class_{cls}",
                                   total_recall_for_each_class[cls]/(num_step * loader.batch_size),
                                   self.training_step)
        self.writer.add_scalar("meanRecall", 
            torch.sum(total_recall_for_each_class[1:5])/(num_step * loader.batch_size * (model.out_channels-2)),
            self.training_step)

        model.train()

    def save_predictions_as_img(self, loader, model):

        model.eval()
        
        x, y = next(iter(loader))
        x = x.to(device=self.device)
        with torch.no_grad():
            preds = torch.argmax(model(x), dim=1, keepdim=False)
            preds_np = preds.cpu().numpy()
 
            for idx in range(preds_np.shape[0]):
                
                #binary_mask = np.expand_dims(np.not_equal(y[idx], 0), axis=2)
                 ##
                #rgb_mask = binary_mask * color_group[preds_np[idx]]
                rgb_mask = color_group[preds_np[idx]]

                self.writer.add_image(f'{idx}/predict', rgb_mask/255., self.training_step, dataformats="HWC")

                rgb_mask_groundtruth = color_group[y[idx]]
                self.writer.add_image(f'{idx}/target', rgb_mask_groundtruth/255., self.training_step, dataformats="HWC")

        model.train()

    def log_step(self, loss):
        self.accumulate_training_loss += loss
        self.epoch_num_step += 1
        self.training_step += 1

    def log_epoch(self, val_loader, model, optimizer):
        self.writer.add_scalar('Loss/Average_training_loss',
                               self.accumulate_training_loss/ self.epoch_num_step, self.training_step)
        
        self.accumulate_training_loss = 0
        self.epoch_num_step = 0

        
        self.save_predictions_as_img(val_loader, model)
        self.validation(val_loader, model)