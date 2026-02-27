import torch
import torch.nn as nn
from torchvision import models

class ResNet18GTSRB(nn.Module):
    def __init__(self, num_classes=43, pretrained=True):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def freeze(self):
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
    