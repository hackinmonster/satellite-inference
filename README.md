# Satellite Imagery Inference Engine

Upload a satellite tile and get a land-cover label back.

The model classifies 64×64 RGB patches as water, forest, urban, agriculture, barren, or cloud. It is a small CNN trained on synthetic tiles that mimic Sentinel-2 colors. FastAPI serves it so you can use the browser UI or curl.

If you have a GPU, concurrent requests get batched together. Redis can cache identical images. TensorRT can export an FP16 engine. CPU still works.

## Quick start

Python 3.10+ and [PyTorch](https://pytorch.org/get-started/locally/). CUDA is used when `torch.cuda.is_available()`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), click a sample tile, then Predict. Predict grid runs a 64×64 window across larger images.

```bash
python infer.py samples/urban.png
curl -F "file=@samples/forest.png" http://127.0.0.1:8000/predict
```

Weights live in `weights/landcover.pt`. Run `python train.py` if you want to regenerate them.

## API


| endpoint              | what it does                                                       |
| --------------------- | ------------------------------------------------------------------ |
| `POST /predict`       | one image                                                          |
| `POST /predict/batch` | several images in one forward pass                                 |
| `POST /predict/grid`  | sliding-window map over a bigger image                             |
| `GET /health`         | device and backend (`pytorch`, `pytorch-fp16`, or `tensorrt-fp16`) |


The UI is `static/index.html` at `/`.

Notes:

- Images are resized to 64×64 and kept in `[0, 1]`. ImageNet mean/std did not match training and broke urban/agriculture predictions.
- Redis is optional. Cache keys are a SHA-256 of the file bytes. Filenames collided when two different images used the same name.
- The batcher waits up to 8 ms to fill a batch of 16. A 50 ms window added too much latency on single requests.

Env vars (see `.env.example`):


| var          | default     |
| ------------ | ----------- |
| `REDIS_HOST` | `localhost` |
| `REDIS_PORT` | `6379`      |
| `REDIS_DB`   | `2`         |


```bash
docker run -d -p 6379:6379 redis:7   # optional
```

## TensorRT

```bash
python export_onnx.py
python export_trt.py
```

`export_trt.py` exits if TensorRT is not installed. When the build works, it writes `weights/landcover.engine` with FP16 if the GPU supports it. `Engine` tries that file on startup and falls back to PyTorch.

## Benchmark

```bash
python benchmark.py --n 128 --batch 16
```

On CPU, batching this CNN is about 1.4×. On GPU, compare sequential FP32 against batched FP16 TensorRT.

## Docker

```bash
docker compose up --build
```

API is at [http://localhost:8000](http://localhost:8000). Compose starts Redis. The Dockerfile installs CPU PyTorch. Use a CUDA base image if you want the GPU stack.

## Tests

```bash
pytest tests/
```

## Layout

```
model.py          CNN + class names
train.py          synthetic data + training
preprocess.py     resize / to-tensor
engine.py         pytorch / tensorrt inference
batcher.py        async request coalescing
cache.py          redis
detect.py         tiled grid
server.py         fastapi
trt_engine.py     tensorrt wrapper
static/           upload page
weights/          landcover.pt, landcover.onnx
```

