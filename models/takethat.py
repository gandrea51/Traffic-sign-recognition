import torch
import torch.nn as nn
import torch.nn.functional as F

class TakeThat(nn.Module):
    def __init__(self, input_channels=3, num_classes=43):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),  # per input 64x64
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x