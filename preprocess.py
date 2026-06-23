import io

import torch
from PIL import Image
from torchvision import transforms

# match training: [0, 1] tensors, no ImageNet mean/std
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


def load_image(source):
    """source can be a path, bytes, or a PIL image."""
    if isinstance(source, bytes):
        img = Image.open(io.BytesIO(source)).convert("RGB")
    elif isinstance(source, Image.Image):
        img = source.convert("RGB")
    else:
        img = Image.open(source).convert("RGB")
    return transform(img)
