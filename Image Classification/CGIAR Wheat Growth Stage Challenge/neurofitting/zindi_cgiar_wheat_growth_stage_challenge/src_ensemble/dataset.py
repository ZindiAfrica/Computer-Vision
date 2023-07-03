from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

class ZCTESTDataset(Dataset):
    def __init__(self, imgs):
        super(ZCTESTDataset).__init__()
        self.imgs = imgs 
        self.transform = A.Compose([ToTensorV2(always_apply=True, p=1.0)], p=1.0)
        self.transform_hflip = A.Compose([
                                    A.HorizontalFlip(p=1.0),
                                    ToTensorV2(always_apply=True, p=1.0)
                                    ], p=1.0)       
    def __len__(self):
        return self.imgs.shape[0]        
    def __getitem__(self, index):
        img = self.imgs[index].copy()
        img = (img/255).astype(np.float32)
        img_hflip = self.transform_hflip(**{'image': img})['image']
        img = self.transform(**{'image': img})['image']
        return img.float(), img_hflip.float()