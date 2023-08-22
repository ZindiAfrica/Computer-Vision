import cv2
import numpy as np
import torch
from ishtos_transforms import get_transforms
from torch.utils.data import Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, config, df, phase="train", transforms=None):
        self.config = config
        self.image_paths = df["image_path"].values
        if phase in ["train", "valid"]:
            self.targets = df[config.dataset.target].values
        self.phase = phase
        self.transforms = transforms
        self.store_train = phase == "train" and config.dataset.store_train
        self.store_valid = phase == "valid" and config.dataset.store_valid

        if self.store_train or self.store_valid:
            self.images = [
                self.load_image(image_path, config)
                for image_path in tqdm(self.image_paths)
            ]

    def __getitem__(self, index):
        if self.store_train or self.store_valid:
            image = self.images[index]
        else:
            image = self.load_image(self.image_paths[index], self.config)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if self.phase in ["train", "valid"]:
            return image, torch.tensor(self.targets[index], dtype=torch.long)
        else:
            return image

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, image_path, config):
        image = cv2.imread(image_path)
        if config.dataset.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, 2)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if config.dataset.gradcam:
            image = cv2.resize(
                image,
                (config.transforms.params.height, config.transforms.params.width),
            )

        return image


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_dataset(config, df, phase, apply_transforms=True):
    transforms = get_transforms(config, phase) if apply_transforms else None
    return MyDataset(config, df, phase, transforms)
