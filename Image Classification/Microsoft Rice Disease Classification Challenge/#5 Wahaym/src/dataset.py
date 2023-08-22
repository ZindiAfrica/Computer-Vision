from argparse import Namespace
from typing import List, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from sklearn import preprocessing
from torch.utils.data import Dataset

from config import Config


def get_augmentations(aug, image_size):
    if (type(aug) is dict):
        aug = Namespace(**aug)

        print("====Aug from dataset===")
        print(aug)

    return {"train": [A.Resize(image_size[0], image_size[1]),
            A.Affine(
        rotate=(-aug.rotate, aug.rotate),
        translate_percent=(0.0, aug.translate),
        shear=(-aug.shear, aug.shear),
        p=aug.p_affine,
    ),
        A.RandomResizedCrop(
                image_size[0],
                image_size[1],
                scale=(aug.crop_scale, 1.0),
                ratio=(aug.crop_l, aug.crop_r),
    ),
        A.ToGray(p=aug.p_gray),
        A.GaussianBlur(blur_limit=(7, 15), p=aug.p_blur),
        A.GaussNoise(p=aug.p_noise),
        A.CLAHE(clip_limit=6.0, tile_grid_size=(10, 10), p=aug.p_clahe),
        A.RGBShift(r_shift_limit=50, g_shift_limit=50,
                   b_shift_limit=50, p=aug.p_rgb_shift),
        A.Downscale(scale_min=0.5, scale_max=0.5, p=aug.p_downscale),
        A.RandomGridShuffle(grid=(2, 2), p=aug.p_shuffle),
        A.Posterize(p=aug.p_posterize),
        A.RandomBrightnessContrast(p=aug.p_bright_contrast),
        A.CoarseDropout(p=aug.p_cutout),
        A.RandomSnow(p=aug.p_snow),
        A.RandomRain(p=aug.p_rain),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5), ],
        "val": [A.Resize(image_size[0], image_size[1])]}


class RiceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        image_dir: str,
        img_format: str,
        data_aug: bool,
    ):
        super().__init__()
        self.index = df.index
        self.x_paths = np.array(df.Image_id)
        self.label = np.array(df.target, dtype=int) if hasattr(
            df, "target") else np.full(len(df), -1)
        self.cfg = cfg
        self.df = df
        self.image_dir = image_dir
        self.img_format = img_format
        self.data_aug = data_aug
        augments = []
        aug = cfg.aug
        if data_aug:

            augments = get_augmentations(
                aug, self.cfg.image_size)["train"]

        else:
            augments = get_augmentations(
                aug, self.cfg.image_size)["val"]

        augments.append(A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        augments.append(ToTensorV2())  # HWC to CHW
        self.transform = A.Compose(augments)

    def __len__(self):
        return len(self.df)

    def get_original_image(self, i: int):
        if self.img_format == "rgn":
            fname = self.x_paths[i][:-4] + "_rgn.jpg"
            bgr = cv2.imread(f"{self.image_dir}/{fname}")
        else:
            bgr = cv2.imread(f"{self.image_dir}/{self.x_paths[i]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def __getitem__(self, i: int):

        image = self.get_original_image(i)

        # data augmentation
        augmented = self.transform(image=image)["image"]
        return {
            "original_index": self.index[i],
            "image": augmented,
            "label": self.label[i],
        }


def load_df(in_base_dir: str, cfg: Config, filename: str,) -> pd.DataFrame:
    filename = cfg.train_csv if filename == "Train" else filename
    df = pd.read_csv(f"{in_base_dir}/{filename}")
    if hasattr(df, "Label"):
        disease_ids = np.sort(df.Label.unique())
        disease_id2class = {lid: i for i, lid in enumerate(disease_ids)}
        df['target'] = df['Label'].apply(lambda x: disease_id2class[x])
        assert cfg.num_classes == len(disease_ids)
    return df
