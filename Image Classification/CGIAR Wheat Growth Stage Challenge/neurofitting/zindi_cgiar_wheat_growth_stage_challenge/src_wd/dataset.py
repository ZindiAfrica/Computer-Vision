import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

def get_train_transforms():      
    return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit= 20, val_shift_limit=20, p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            ],p=0.5), 
            A.JpegCompression (quality_lower=90, quality_upper=100, always_apply=False, p=0.5),
            A.RandomSizedCrop(min_max_height=(164, 256), height=256, width=512, p=0.5),            
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=None, always_apply=False, p=0.5),                                                    
            A.Cutout(num_holes=8, max_h_size=28, max_w_size=28, fill_value=0, p=0.5),
            A.HorizontalFlip(p=0.5),
        ], p=1.0)

class ZCDataset(Dataset):
    def __init__(self, imgs, labels, test):
        super(ZCDataset).__init__()
        self.imgs = imgs
        self.labels = labels
        self.test = test
        
        if not test:
            self.transform = get_train_transforms()
        
    def __len__(self):
        return self.imgs.shape[0]
        
    def __getitem__(self, index):
        label = torch.tensor(self.labels[index]).float() #long()
        img = self.imgs[index].copy()
        
        if not self.test:
            img = self.transform(**{'image': img})['image']
        
        img = torch.tensor(img).float()
        img = img/255
        img = img.permute([2,0,1])
        return img, label