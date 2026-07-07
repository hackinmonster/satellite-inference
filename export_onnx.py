import argparse

import torch

from model import LandCoverNet


def export(weights="weights/landcover.pt", out="weights/landcover.onnx"):
    model = LandCoverNet()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # torch 2.9 wants onnxscript otherwise
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/landcover.pt")
    parser.add_argument("--out", default="weights/landcover.onnx")
    args = parser.parse_args()
    export(args.weights, args.out)
