from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

def data_transforms(phase='train', img_size=512):
    if phase == 'train':
        return A.Compose([
            
            A.RandomBrightness(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Resize(img_size, img_size),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ),
            ToTensorV2()])
        