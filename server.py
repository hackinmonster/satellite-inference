import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from infer import load_model
from model import CLASSES
from preprocess import load_image

app = FastAPI(title="Satellite Inference")

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def startup():
    global model
    model = load_model("weights/landcover.pt", device)
    print(f"model loaded on {device}")


@app.get("/health")
def health():
    return {"ok": True, "device": str(device)}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(500, "model not loaded")

    try:
        x = load_image(file.file.read()).unsqueeze(0).to(device)
    except Exception:
        raise HTTPException(400, "could not read image")

    start = time.time()
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
