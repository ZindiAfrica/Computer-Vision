from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
from scipy import ndimage
import torch
from PIL import Image

class ICLRDataset(Dataset):
    def __init__(self, imgs, gts, split_type, index, transform, img_mix_enable = True):
        if index is None:
            self.imgs = imgs
            self.gts = gts
        else:
            self.imgs = [imgs[i] for i in index]
            self.gts = [gts[i] for i in index] 
                   
        self.split_type = split_type
        self.transform = transform
        self.img_mix_enable = img_mix_enable
    
    def __len__(self):
        return len(self.imgs)
    
    def augment(self, img, y):        
        p = np.random.random(1)
        if p[0] > 0.5:
            while True:
                rnd_idx = np.random.randint(0, len(self.imgs))
                if self.gts[rnd_idx] != y:
                    break
            rnd_crop = self.transform(Image.fromarray(self.imgs[rnd_idx]))
            d = 0.8
            img = img * d + rnd_crop * (1 - d)
        return img

    def __getitem__(self, idx):
        img = self.imgs[idx]
        y = self.gts[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        if (self.split_type == 'train') & self.img_mix_enable:
            img = self.augment(img, y)
        return img, y

