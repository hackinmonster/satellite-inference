import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import cache
from batcher import Batcher
from detect import classify_grid
from engine import Engine
from preprocess import load_image

engine = None
batcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, batcher
    engine = Engine()
    batcher = Batcher(engine)
    batcher.start()
    cache.connect()
    print(f"backend={engine.backend} device={engine.device}")
    yield
    if batcher is not None and batcher.task is not None:
        batcher.task.cancel()


app = FastAPI(title="Satellite Inference", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(engine.device) if engine else None,
        "backend": engine.backend if engine else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if batcher is None:
        raise HTTPException(500, "model not loaded")
    data = await file.read()
    cached = cache.get(data)
    if cached:
        cached["cached"] = True
        return cached
    try:
        result = await batcher.infer(data)
    except UnidentifiedImageError:
        raise HTTPException(400, "could not read image")
    cache.put(data, result)
    result["cached"] = False
    return result


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
    except UnidentifiedImageError:
        raise HTTPException(400, "could not read image")
    results, latency_ms = engine.predict_many(tensors)
    return {"results": results, "batch_latency_ms": round(latency_ms, 2), "n": len(results)}


@app.post("/predict/grid")
def predict_grid(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(500, "model not loaded")
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "could not read image")
    try:
        return classify_grid(engine, img)
    except ValueError as e:
        raise HTTPException(400, str(e))


app.mount("/samples", StaticFiles(directory="samples"), name="samples")
