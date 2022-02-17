import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms():      
    return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            ],p=0.5),
            A.JpegCompression(quality_lower=90, quality_upper=100, always_apply=False, p=0.5),
            A.HorizontalFlip(p=0.5), 
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)

def get_val_transforms():      
    return A.Compose([
            ToTensorV2(always_apply=True, p=1.0),
        ], p=1.0)
       
class ZCDataset(Dataset):
    def __init__(self, imgs, labels, transform, test):
        super(ZCDataset).__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.test = test        
    def __len__(self):
        return self.imgs.shape[0]        
    def __getitem__(self, index):
        label = torch.tensor(self.labels[index])
        img = self.imgs[index].copy()
        img = (img/255).astype(np.float32)
        img = self.transform(**{'image': img})['image']
        return img.float(), label.float()