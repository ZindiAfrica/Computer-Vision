import timm
import torch.nn as nn


class NeckV1(nn.Module):
    def __init__(self, in_features, out_features):
        super(HeadV1, self).__init__()
        self.head = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.head(x)


class HeadV1(nn.Module):
    def __init__(self, in_features, out_features):
        super(HeadV1, self).__init__()
        self.head = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.head(x)


# --------------------------------------------------
# ResNet
# - resnet18, 26, 34, 50, 101, 152, 200
# - resnet18d, 26, 34, 50, 101, 152, 200
# --------------------------------------------------

# --------------------------------------------------
# ConvNeXt
# - convnext_tiny, small, base, large
# - convnext_base_in22ft1k, large, xlarge
# - convnext_base_384_in22ft1k, large, xlarge
# - convnext_base_in22k, large, xlarge
# --------------------------------------------------

# --------------------------------------------------
# EfficientNet
# - efficientnet_b0 ~ b4
# - efficientnet_es, m, l
# - efficientnet_es_pruned, l
# - efficientnet_b1_pruned ~ b3
# - efficientnetv2_rw_t, s, m
# - tf_efficientnet_b0 ~ b8
# - tf_efficientnet_b0_ap ~ b8
# - tf_efficientnet_b0_ns ~ b7
# - tf-efficientnet_es, m, l
# - tf_efficientnetv2_s, m, l
# - tf_efficientnetv2_s_in21k, m, l, xl
# - tf_efficientnetv2_b0 ~ b3
# --------------------------------------------------

# --------------------------------------------------
# SwinTransformer
# - swin_base_patch4_window12_384, large
# - swin_base_patch4_window7_224, tiny, small, large
# - swin_base_patch4_window12_384_in22k, large
# - swin_base_patch4_window7_224_in22k, large
# --------------------------------------------------
class Net(nn.Module):
    def __init__(
        self,
        base_model="swin_tiny_patch4_window7_224",
        pretrained=True,
        checkpoint_path=None,
        num_classes=2,
        neck_version="v1",
        head_version="v1",
    ):
        super(Net, self).__init__()

        self.backbone = timm.create_model(
            base_model, pretrained=pretrained, checkpoint_path=checkpoint_path
        )
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(num_classes=0, global_pool="avg")
        self.neck = get_neck(
            version=neck_version,
            in_features=in_features,
            out_features=in_features,
        )
        self.head = get_head(
            version=head_version, in_features=in_features, out_features=num_classes
        )

    def forward(self, x):
        x = self.backbone(x)
        if self.neck:
            x = self.neck(x)
        if self.head:
            x = self.head(x)
        return x


# --------------------------------------------------
# getter
# --------------------------------------------------
def get_model(config):
    model_name = config.name
    if model_name == "net":
        return Net(**config.params)
    else:
        raise ValueError(f"Not supported model: {model_name}")


def get_neck(version, in_features, out_features):
    if version is None:
        return None
    elif version == "v1":
        return NeckV1(in_features=in_features, out_features=out_features)
    else:
        raise ValueError(f"Not supported head version: {version}")


def get_head(version, in_features, out_features):
    if version is None:
        return None
    elif version == "v1":
        return HeadV1(in_features=in_features, out_features=out_features)
    else:
        raise ValueError(f"Not supported head version: {version}")
