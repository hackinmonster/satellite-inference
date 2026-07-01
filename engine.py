import time

import torch
import torch.nn.functional as F

from model import CLASSES, LandCoverNet
from preprocess import load_image


class Engine:
    def __init__(self, weights="weights/landcover.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LandCoverNet()
        state = torch.load(weights, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.fp16 = self.device.type == "cuda"
        if self.fp16:
            self.model.half()
        self.model.to(self.device)
        self.model.eval()
        self.backend = "pytorch-fp16" if self.fp16 else "pytorch"

    def _run(self, batch):
        if self.fp16:
            batch = batch.half()
        batch = batch.to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits.float(), dim=1)
        return probs.cpu()

    def predict(self, image_bytes):
        x = load_image(image_bytes).unsqueeze(0)
        t0 = time.perf_counter()
        probs = self._run(x)[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        idx = int(probs.argmax())
        return {
            "label": CLASSES[idx],
            "confidence": round(float(probs[idx]), 4),
            "scores": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
            "latency_ms": round(latency_ms, 2),
            "backend": self.backend,
        }

    def predict_many(self, tensors):
        batch = torch.stack(tensors)
        t0 = time.perf_counter()
        probs = self._run(batch)
        latency_ms = (time.perf_counter() - t0) * 1000
        out = []
        for p in probs:
            idx = int(p.argmax())
            out.append(
                {
                    "label": CLASSES[idx],
                    "confidence": round(float(p[idx]), 4),
                    "scores": {
                        CLASSES[i]: round(float(p[i]), 4) for i in range(len(CLASSES))
                    },
                }
            )
        return out, latency_ms
