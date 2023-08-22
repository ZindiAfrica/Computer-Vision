import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms_v1(config, pretrained):
    augmentations = [A.Resize(config.height, config.width)]
    if pretrained:
        augmentations.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(A.Normalize(mean=0, std=1))
    augmentations.append(ToTensorV2())
    return A.Compose(augmentations)


def get_train_transforms_v2(config, pretrained):
    augmentations = [
        A.HorizontalFlip(p=config.p),
        A.VerticalFlip(p=config.p),
        A.ImageCompression(quality_lower=99, quality_upper=100),
        A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=config.p
        ),
        A.Resize(config.height, config.width),
        A.Cutout(
            max_h_size=int(config.height * 0.4),
            max_w_size=int(config.width * 0.4),
            num_holes=1,
            p=config.p,
        ),
    ]
    if pretrained:
        augmentations.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(A.Normalize(mean=0, std=1))
    augmentations.append(ToTensorV2())
    return A.Compose(augmentations)


def get_valid_transforms_v1(config, pretrained):
    augmentations = [A.Resize(config.height, config.width)]
    if pretrained:
        augmentations.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        augmentations.append(A.Normalize(mean=0, std=1))
    augmentations.append(ToTensorV2())
    return A.Compose(augmentations)


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_transforms(config, phase):
    try:
        if phase == "train":
            return eval(f"get_train_transforms_{config.transforms.train_version}")(
                config.transforms.params,
                config.model.params.pretrained,
            )
        elif phase in ["valid", "test"]:
            return eval(f"get_valid_transforms_{config.transforms.valid_version}")(
                config.transforms.params,
                config.model.params.pretrained,
            )
        else:
            raise ValueError(f"Not supported transforms phase: {phase}.")
    except NameError:
        return None
