from typing import Tuple, Dict, Optional, Callable, Any, Sequence

import albumentations as albu
import cv2
import numpy as np
import random
import torch
from torch.utils import data as torch_data


class ZindiWheatDataset(torch_data.Dataset):
    """Custom dataset for Zindi Wheat competition."""

    def __init__(
        self,
        images: Sequence[str],
        labels: Optional[Sequence[int]] = None,
        preprocess_function: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        augmentations: Optional[albu.Compose] = None,
        input_shape: Tuple[int, int, int] = (128, 128, 3),
        crop_method: str = "resize",
        augment_label: float = 0.0,
    ) -> None:
        """
        Args:
            images: sequence of input images
            labels: sequence of corresponding labels
            preprocess_function: normalization function for input images
            augmentations: list of augmentation to be applied
            input_shape: image input shape to the model
            crop_method: one of {'resize', 'crop'}. Cropping strategy for input images
                - 'resize' corresponds to resizing the image to the input shape
                - 'crop' corresponds to random cropping from the given image
            augment_label: probability to perform label augmentation
        """
        self.images = images
        self.labels = labels
        self.preprocess_function = preprocess_function
        self.augmentations = augmentations
        self.input_shape = input_shape
        self.crop_method = crop_method
        self.augment_label = augment_label

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, image_index: int) -> Dict[str, Any]:
        sample = dict()

        sample["image"] = self._read_image(image_index)

        if self.labels is not None:
            sample = self._read_label(image_index, sample)

        if self.crop_method is not None:
            sample = self._crop_data(sample)

        if self.augmentations is not None:
            sample = self._augment_data(sample)

        if self.preprocess_function is not None:
            sample = self._preprocess_data(sample)

        return sample

    def _read_image(self, image_index: int) -> np.ndarray:
        img = cv2.imread(self.images[image_index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_label(self, image_index: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        aug_label = self.labels[image_index]

        # Label augmentation assigns neighbor classes with the probability self.augment_label
        if self.augment_label > 0:
            p = random.random()
            if p < self.augment_label / 2:
                aug_label = max(0, aug_label - 1)
            elif p < self.augment_label:
                aug_label = min(max(self.labels), aug_label + 1)

        sample["label"] = aug_label
        return sample

    def _crop_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.crop_method == "resize":
            aug = albu.Compose(
                [
                    albu.PadIfNeeded(
                        min_height=sample["image"].shape[1] // 2,
                        min_width=sample["image"].shape[1],
                        border_mode=cv2.BORDER_CONSTANT,
                        always_apply=True,
                    ),
                    albu.Resize(
                        height=self.input_shape[0],
                        width=self.input_shape[1],
                        interpolation=cv2.INTER_LINEAR,
                        always_apply=True,
                    ),
                ]
            )
        elif self.crop_method == "crop":
            aug_list = [
                albu.PadIfNeeded(
                    min_height=sample["image"].shape[1] // 2,
                    min_width=sample["image"].shape[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    always_apply=True,
                ),
                albu.Resize(
                    height=self.input_shape[0],
                    width=self.input_shape[1] * 2,
                    interpolation=cv2.INTER_LINEAR,
                    always_apply=True,
                ),
            ]

            if self.labels is not None:
                aug_list.append(
                    albu.RandomCrop(
                        height=self.input_shape[0],
                        width=self.input_shape[1],
                        always_apply=True,
                    )
                )
            aug = albu.Compose(aug_list)
        else:
            raise ValueError(f"{self.crop_method} cropping strategy is not available")

        return aug(**sample)

    def _augment_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = self.augmentations(**sample)
        return sample

    def _preprocess_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image"] = self.preprocess_function(sample["image"])
        return sample
