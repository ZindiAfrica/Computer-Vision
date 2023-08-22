import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def get_train_transforms(DIM=512):
    return A.Compose(
        [
            A.RandomResizedCrop(height=DIM, width=DIM),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Cutout(always_apply=True, num_holes=32, ),
            A.ShiftScaleRotate(),

            A.Normalize(),

        ]
    )


transform = get_train_transforms()

image = cv2.imread('/home/mithil/PycharmProjects/Rice/data/images/id_0l8i75umhc.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = transform(image=image)['image']
plt.imshow(image)
plt.show()
