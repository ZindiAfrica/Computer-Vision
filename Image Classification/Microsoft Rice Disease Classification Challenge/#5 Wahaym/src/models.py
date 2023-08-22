
import math

# from pytorchcv.model_provider import get_model as ptcv_get_model
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from utils import *


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        # import pdb; pdb.set_trace()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(
                    1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    # last feature if backbone outputs list/tuple of features
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = (feature_size, feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (
            feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(HybridNet, self).__init__()
        self.backbone = timm.create_model(args.backbone, pretrained=pretrained)
        self.embedder = timm.create_model(
            args.embedder, features_only=True, out_indices=[2], pretrained=pretrained)
        self.backbone.patch_embed = HybridEmbed(
            self.embedder, img_size=args.image_size[0], embed_dim=args.embedding_size)
        self.n_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
        self.fc = nn.Linear(self.n_features, args.num_classes)
        if args.pretrained_weights is not None:
            self.load_state_dict(torch.load(
                args.pretrained_weights, map_location='cpu'), strict=False)
            print('weights loaded from', args.pretrained_weights)

    def forward(self, x):
        features = self.backbone(x)
        output = self.fc(features)
        return output


class SimpleNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(SimpleNet, self).__init__()
        self.backbone = timm.create_model(args.backbone, pretrained=pretrained)

        if 'regnet' in args.backbone:
            self.out_features = self.backbone.head.fc.in_features
        if 'nfnet' in args.backbone:
            self.out_features = self.backbone.head.fc.in_features
        elif 'cait' in args.backbone:
            self.out_features = self.backbone.head.in_features
        elif 'xcit' in args.backbone:
            self.out_features = self.backbone.head.in_features
        elif 'convnext' in args.backbone:
            self.out_features = self.backbone.head.fc.in_features
        elif 'beit' in args.backbone:
            self.out_features = self.backbone.head.in_features
        elif 'swin' in args.backbone:
            self.out_features = self.backbone.head.in_features
        elif 'csp' in args.backbone:
            self.out_features = self.backbone.head.fc.in_features
        elif 'res' in args.backbone:  # works also for resnest
            self.out_features = self.backbone.fc.in_features
        elif 'efficientnet' in args.backbone:
            self.out_features = self.backbone.classifier.in_features
        elif 'densenet' in args.backbone:
            self.out_features = self.backbone.classifier.in_features
        elif 'senet' in args.backbone:
            self.out_features = self.backbone.fc.in_features
        elif 'inception' in args.backbone:
            self.out_features = self.backbone.last_linear.in_features

        else:
            self.out_features = self.backbone.head.in_features

        self.backbone.reset_classifier(0)
        if args.grad_checkpointing:
            self.backbone.set_grad_checkpointing()
        self.fc = nn.Linear(self.out_features, args.num_classes)
        if args.pretrained_weights is not None:
            self.load_state_dict(torch.load(
                args.pretrained_weights, map_location='cpu'), strict=False)
            print('weights loaded from', args.pretrained_weights)

    def forward(self, x):

        features = self.backbone(x)  # [:, 0, :]
        output = self.fc(features)
        return output


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EffGeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=p_trainable)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x.clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)


class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()

        self.d0 = nn.Conv2d(in_chans, out_chans//2, kernel_size=3,
                            dilation=dilations[0], padding='same')
        self.d1 = nn.Conv2d(in_chans, out_chans//2, kernel_size=3,
                            dilation=dilations[1], padding='same')
        self.d2 = nn.Conv2d(in_chans, out_chans//2, kernel_size=3,
                            dilation=dilations[2], padding='same')
        self.conv1 = nn.Conv2d(out_chans//2 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x


class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, 1, 1)
        self.bn = nn.BatchNorm2d(in_c)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_c, 1, 1, 1)
        # use default setting.
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.bn(x)

        feature_map_norm = F.normalize(x, p=2, dim=1)

        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)

        x = att * feature_map_norm
        return x, att_score


class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):

        bs, c, w, h = fl.shape

        fl_dot_fg = torch.bmm(fg[:, None, :], fl.reshape(bs, c, -1))
        fl_dot_fg = fl_dot_fg.reshape(bs, 1, w, h)
        fg_norm = torch.norm(fg, dim=1)

        fl_proj = (fl_dot_fg / fg_norm[:, None,
                   None, None]) * fg[:, :, None, None]
        fl_orth = fl - fl_proj

        f_fused = torch.cat(
            [fl_orth, fg[:, :, None, None].repeat(1, 1, w, h)], dim=1)
        return f_fused


class Dlog(nn.Module):
    def __init__(self, args, pretrained=True):
        super(Dlog, self).__init__()

        self.args = args
        self.n_classes = self.args.num_classes
        self.backbone = timm.create_model(args.backbone,
                                          pretrained=pretrained,
                                          num_classes=0,
                                          global_pool="",
                                          in_chans=3, features_only=True)
        if args.grad_checkpointing:
            self.backbone.set_grad_checkpointing()
        if ("efficientnet" in args.backbone) & (self.args.stride is not None):
            self.backbone.conv_stem.stride = self.args.stride
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']

        feature_dim_l_g = self.args.embedding_size * 2
        fusion_out = 2 * feature_dim_l_g

        if args.pool == "gem":
            self.global_pool = GeM(p_trainable=args.gem_p_trainable)
        elif args.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif args.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = args.embedding_size
        self.fc = nn.Linear(fusion_out, args.num_classes)
        self.mam = MultiAtrousModule(
            backbone_out_1, feature_dim_l_g, self.args.dilations)
        self.conv_g = nn.Conv2d(backbone_out, feature_dim_l_g, kernel_size=1)
        self.bn_g = nn.BatchNorm2d(
            feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g = nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()

    def forward(self, x):
        x = self.backbone(x)

        x_l = x[-2]
        x_g = x[-1]

        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)

        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)
        x_g = self.global_pool(x_g)
        x_g = x_g[:, :, 0, 0]

        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_emb = x_fused[:, :, 0, 0]
        output = self.fc(x_emb)
        return output


class EffNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model(
            args.backbone,
            in_chans=3,
            num_classes=0,
            features_only=True,
            out_indices=args.out_indices,
        )
        for name, child in (self.backbone.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
        feature_dims = self.backbone.feature_info.channels()
        print(f"feature dims: {feature_dims}")
        self.global_pools = torch.nn.ModuleList(
            [EffGeM(p=args.global_pool.p, p_trainable=args.global_pool.train)
             for _ in args.out_indices]
        )
        self.mid_features = np.sum(feature_dims)
        if args.normalization == "batchnorm":
            self.neck = torch.nn.BatchNorm1d(self.mid_features)
        elif args.normalization == "layernorm":
            self.neck = torch.nn.LayerNorm(self.mid_features)
        self.fc = nn.Linear(self.mid_features, args.num_classes)
        if args.pretrained_weights is not None:
            self.load_state_dict(torch.load(
                args.pretrained_weights, map_location='cpu'), strict=False)
            print('weights loaded from', args.pretrained_weights)

    def forward(self, x):
        ms = self.backbone(x)
        h = torch.cat([global_pool(m)
                      for m, global_pool in zip(ms, self.global_pools)], dim=1)
        h = self.neck(h)

        output = self.fc(h)
        return output
