import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from engine import Engine
from preprocess import load_image

app = FastAPI(title="Satellite Inference")
engine = None


@app.on_event("startup")
def startup():
    global engine
    engine = Engine()
    print(f"backend={engine.backend} device={engine.device}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(engine.device) if engine else None,
        "backend": engine.backend if engine else None,
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(500, "model not loaded")
    try:
        data = file.file.read()
        load_image(data)
    except Exception:
        raise HTTPException(400, "could not read image")
    return engine.predict(data)


@app.post("/predict/batch")
def predict_batch(files: list[UploadFile] = File(...)):
    if engine is None:
        raise HTTPException(500, "model not loaded")
    if not files:
        raise HTTPException(400, "no files")
    tensors = []
    try:
        for f in files:
            tensors.append(load_image(f.file.read()))
    except Exception:
        raise HTTPException(400, "could not read image")
    results, latency_ms = engine.predict_many(tensors)
    return {"results": results, "batch_latency_ms": round(latency_ms, 2), "n": len(results)}
