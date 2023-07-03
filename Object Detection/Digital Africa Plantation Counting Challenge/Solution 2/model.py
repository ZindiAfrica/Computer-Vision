import torch
import torch.nn as nn
import timm

class PlantationModel(nn.Module):
    def __init__(self, model, pretrained=True):
        super(PlantationModel, self).__init__()
        self.model = timm.create_model(model, pretrained=pretrained)
        self.fc = nn.Linear(self.model.classifier.out_features, 1)
        
    def forward(self, image):
        x = self.model(image)
        x = self.fc(x)
        return x