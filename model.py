import torch
import torch.nn as nn
from torchvision.models import resnet18

CLASSES = ["water", "forest", "urban", "agriculture", "barren", "cloud"]


def LandCoverNet(num_classes=len(CLASSES)):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
