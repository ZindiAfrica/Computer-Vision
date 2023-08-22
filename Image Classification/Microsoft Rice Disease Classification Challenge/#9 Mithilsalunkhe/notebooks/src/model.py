import timm
import torch.nn as nn
import torch
import torch.nn.functional as F


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class BaseModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg['model'], pretrained=self.cfg['pretrained'],
                                       in_chans=self.cfg['in_channels'],
                                       num_classes=cfg['target_size'])
        self.model = self.model.apply(set_batchnorm_eval)

    def forward(self, x):
        output = self.model(x)

        return output


class BaseModelFeature(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg['model'], pretrained=self.cfg['pretrained'],
                                       in_chans=self.cfg['in_channels'],
                                       num_classes=cfg['target_size'])
        self.model = self.model.apply(set_batchnorm_eval)

    def forward(self, x):
        output = self.model(x)

        return output


class myModel(nn.Module):
    def __init__(self,
                 cfg,
                 pretrained=True,
                 multi_drop_rate=0.5,
                 att_layer=True,
                 att_pattern="A"
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.att_layer = att_layer

        self.model = timm.create_model(
            cfg['model'], pretrained=pretrained
        )
        self.model.set_grad_checkpointing(True)
        self.model = self.model.apply(set_batchnorm_eval)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()

        self.head = nn.Linear(n_features, 3)
        self.head_drops = nn.ModuleList()
        for i in range(5):
            self.head_drops.append(nn.Dropout(multi_drop_rate))

        if att_layer:
            if att_pattern == "A":
                self.att_layer = nn.Sequential(
                    nn.Linear(n_features, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
            elif att_pattern == "B":
                self.att_layer = nn.Linear(n_features, 1)
            else:
                raise ValueError("invalid att pattern")

    def forward(self, x):
        if self.att_layer:
            l = x.shape[2] // 2
            h1 = self.model(x[:, :, :l, :l])
            h2 = self.model(x[:, :, :l, l:])
            h3 = self.model(x[:, :, l:, :l])
            h4 = self.model(x[:, :, l:, l:])
            w = F.softmax(torch.cat([
                self.att_layer(h1),
                self.att_layer(h2),
                self.att_layer(h3),
                self.att_layer(h4),
            ], dim=1), dim=1)
            h = h1 * w[:, 0].unsqueeze(-1) + \
                h2 * w[:, 1].unsqueeze(-1) + \
                h3 * w[:, 2].unsqueeze(-1) + \
                h4 * w[:, 3].unsqueeze(-1)
        else:
            h = self.model(x)

        if True:
            for i, dropout in enumerate(self.head_drops):
                if i == 0:
                    output = self.head(dropout(h))
                else:
                    output += self.head(dropout(h))
            output /= len(self.head_drops)
        else:
            output = self.head(h)
        return output
