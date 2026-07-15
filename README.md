# Satellite Imagery Inference Engine

Classifies small RGB satellite tiles into 6 land-cover classes:

`water`, `forest`, `urban`, `agriculture`, `barren`, `cloud`

The model is a small CNN trained on synthetic 64×64 tiles (color/texture stand-ins for real Sentinel-2 patches). A FastAPI service runs inference with optional GPU batching, Redis caching, and a TensorRT FP16 path when an NVIDIA GPU is available.

## Features

- `POST /predict` — one image
- `POST /predict/batch` — several images in one GPU/CPU batch
- `POST /predict/grid` — sliding-window map over a larger image
- Redis cache keyed by SHA-256 of the image bytes
- Concurrent requests are coalesced into batches (8 ms window, up to 16)
- ONNX export + TensorRT engine build (`export_onnx.py`, `export_trt.py`)
- Simple browser UI at `/`

## Setup

Needs Python 3.10+ and [PyTorch](https://pytorch.org/get-started/locally/). CPU works. CUDA is used automatically if `torch.cuda.is_available()`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```

Redis is optional. If nothing is listening, the API still runs and just skips the cache.

```bash
# optional
docker run -d -p 6379:6379 redis:7
```

Environment variables (see `.env.example`):

| var | default | meaning |
|---|---|---|
| `REDIS_HOST` | `localhost` | redis hostname |
| `REDIS_PORT` | `6379` | redis port |
| `REDIS_DB` | `2` | redis database number |

## Run

```bash
python train.py          # regenerate weights (optional, checkpoint is in weights/)
python infer.py samples/urban.png
uvicorn server:app --reload --port 8000
```

Open http://127.0.0.1:8000 and click a sample tile, then Predict or Predict grid.

```bash
curl -F "file=@samples/forest.png" http://127.0.0.1:8000/predict
```

## TensorRT

```bash
python export_onnx.py
python export_trt.py          # no-ops if TensorRT isn't installed
```

`export_trt.py` builds `weights/landcover.engine` with FP16 when the GPU supports it. On startup, `Engine` tries that file first and falls back to PyTorch (FP16 on CUDA, FP32 on CPU).

## Benchmark

```bash
python benchmark.py --n 128 --batch 16
```

On this CPU-only machine the small CNN is already cheap, so batching is only a modest win (~1.4×). The larger speedup is on GPU: FP16 TensorRT plus batched inference versus naive sequential FP32. Re-run the script on a CUDA box after `export_trt.py` to compare backends.

## Docker

```bash
docker compose up --build
```

API is on http://localhost:8000. Compose starts Redis internally. The image installs CPU PyTorch; for GPU/TensorRT use a CUDA base image and skip the CPU index in the Dockerfile.

## Tests

```bash
pytest tests/
```

## Layout

```
model.py          CNN + class names
train.py          synthetic data + training loop
preprocess.py     resize / to-tensor (must match training)
engine.py         pytorch / tensorrt inference
batcher.py        async request coalescing
cache.py          redis
detect.py         tiled grid over larger images
server.py         fastapi
trt_engine.py     tensorrt runtime wrapper
static/           upload page
weights/          landcover.pt and landcover.onnx
```

The first training run saved the whole `nn.Module` pickle and inference used ImageNet mean/std, which did not match training. Both of those are gone; checkpoints are `state_dict`s and tensors stay in `[0, 1]`.
