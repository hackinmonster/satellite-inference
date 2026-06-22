import io
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from model import CLASSES

app = FastAPI(title="Satellite Inference")

preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_model():
    global model
    model = torch.load("weights/landcover.pt", map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    print(f"model loaded on {device}")


@app.get("/health")
def health():
    return {"ok": True, "device": str(device)}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(500, "model not loaded")

    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not read image")

    start = time.time()
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
    elapsed_ms = (time.time() - start) * 1000

    return {
        "label": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
        "scores": {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))},
        "latency_ms": round(elapsed_ms, 2),
    }
