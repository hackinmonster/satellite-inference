import argparse

import torch

from model import CLASSES, LandCoverNet
from preprocess import load_image


def load_model(path="weights/landcover.pt", device="cpu"):
    model = LandCoverNet()
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "conv1.weight" in state:
        model.load_state_dict(state)
    else:
        # old checkpoints saved the whole module
        model = state
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, device="cpu"):
    x = load_image(image_path).unsqueeze(0).to(device)
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
