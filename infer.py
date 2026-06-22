import argparse
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import CLASSES

# copied from a torchvision tutorial — might not match training
preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(path="weights/landcover.pt"):
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def predict(model, image_path):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]), probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--weights", default="weights/landcover.pt")
    args = parser.parse_args()

    model = load_model(args.weights)
    label, conf, _ = predict(model, args.image)
    print(f"{label}  ({conf:.3f})")


if __name__ == "__main__":
    main()
