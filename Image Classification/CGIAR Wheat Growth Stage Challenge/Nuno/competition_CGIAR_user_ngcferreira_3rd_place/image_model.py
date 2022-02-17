import torch
import torch.nn as nn
from base_model import BaseModel
from pytorchcv.model_provider import get_model as ptcv_get_model


class ImageModel(BaseModel):
    @staticmethod
    def get_outputs_from_layer(layer):
        children = list(layer.children())
        if len(children) > 0:
            layer = children[0]
        if isinstance(layer, nn.Conv2d):
            return layer.in_channels
        else:
            return layer.in_features

    def __init__(self, model_name: str, device: torch.device, dropout: float = 0.2, neurons: int = 128,
                 num_classes: int = 1, extras_inputs: list = [], base_model_pretrained_weights: str = None):
        super(ImageModel, self).__init__()
        self.model_name = model_name
        self.device = device
        self.dropout = dropout
        self.neurons = neurons
        self.extras_inputs = extras_inputs

        pretrained = False if base_model_pretrained_weights else True
        self.model = ptcv_get_model(model_name, pretrained=pretrained)
        last_layer = self.model.output

        if base_model_pretrained_weights:
            state_dict = torch.load(base_model_pretrained_weights, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)

        if isinstance(last_layer, nn.Sequential):
            children = list(last_layer)
        else:
            children = []

        total_outputs = self.get_outputs_from_layer(layer=children[-1] if children else last_layer)

        self.original_model_outputs = total_outputs

        self.model.output = nn.Identity()

        # self.model.features.final_pool = nn.AdaptiveAvgPool2d(2048)  # This is needed, but isn't the best option

        if extras_inputs:
            self.extras_inputs_layer = nn.Linear(len(extras_inputs), neurons)
            total_outputs += neurons

        if neurons > 0:
            classifier_layer = [
                nn.Linear(total_outputs, neurons),
                nn.ReLU(),
                nn.BatchNorm1d(neurons),
                nn.Dropout(dropout),

                nn.Linear(neurons, neurons),
                nn.ReLU(),
                nn.BatchNorm1d(neurons),
                nn.Dropout(dropout),

                nn.Linear(neurons, num_classes)
            ]
        else:
            classifier_layer = [
                nn.Dropout(dropout),
                nn.Linear(total_outputs, num_classes)
            ]

        self.last_linear = nn.Sequential(*classifier_layer)
        self.to(device)

    def forward(self, image, extras=None) -> torch.Tensor:
        x = image.to(self.device)
        x = self.model(x)
        x = self.last_linear(x)
        return x

    def forward_data_batch(self, data_batch) -> torch.Tensor:
        extras = None
        image = data_batch["images"]

        if self.extras_inputs:
            batch_size = len(data_batch["images"])
            extras = []
            for key in data_batch.keys():
                if key not in ['images', 'targets']:
                    extras.append(data_batch[key].reshape(batch_size, 1))
            extras = torch.cat(extras, axis=1)

        return self.forward(image=image, extras=extras)

    def input_info(self):
        return dict(input_size=self.model.input_size, mean=self.model.mean, std=self.model.std)
