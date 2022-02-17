import os
import numpy
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset


class WheatDataset(Dataset):
    def __init__(self, df, config, tile_mode: int = 0, rand: bool = False, resize_transform: callable = None,
                 transform: callable = None):
        self.df = df.reset_index(drop=True)
        self.is_train = 'growth_stage' in df.columns
        self.num_classes = config.num_classes
        self.image_dir = config.images_dir
        self.is_regression = config.regression
        self.rand = rand
        self.resize_transform = resize_transform
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row.UID + '.jpeg'

        image_path = os.path.join(self.image_dir, filename)
        images = numpy.asarray(Image.open(image_path).convert('RGB'))

        if self.transform is not None:
            images = self.transform(image=images)['image']

        if self.resize_transform is not None:
            images = self.resize_transform(images)

        images = images.astype(numpy.float32)
        images /= 255
        images = images.transpose((2, 0, 1))
        images = torch.tensor(images)

        if self.is_train:
            if self.is_regression:
                label = row.growth_stage
            else:
                label = numpy.zeros(self.num_classes)
                label[row.growth_stage - 1] = 1

            return images, torch.tensor(label.astype(numpy.float32))
        else:
            return images
