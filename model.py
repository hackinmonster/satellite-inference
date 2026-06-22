import torch
import torch.nn as nn


CLASSES = ["water", "forest", "urban", "agriculture", "barren", "cloud"]


class LandCoverNet(nn.Module):
    """Small CNN for 64x64 RGB satellite tiles."""

    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 16 * 16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
