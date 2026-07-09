import argparse
import io
import time

import torch

from engine import Engine
from preprocess import load_image
from train import make_tile


def fake_bytes(label="urban"):
    buf = io.BytesIO()
    make_tile(label).save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    engine = Engine()
    images = [fake_bytes(["water", "forest", "urban", "agriculture"][i % 4]) for i in range(args.n)]
    tensors = [load_image(b) for b in images]

    # warmup
    engine._run(tensors[0].unsqueeze(0))

    t0 = time.perf_counter()
    for t in tensors:
        engine._run(t.unsqueeze(0))
    sequential_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(0, args.n, args.batch):
        chunk = tensors[i : i + args.batch]
        engine._run(torch.stack(chunk))
    batched_s = time.perf_counter() - t0

    seq_ips = args.n / sequential_s
    bat_ips = args.n / batched_s
    print(f"backend: {engine.backend}  device: {engine.device}")
    print(f"sequential  {args.n} images: {sequential_s*1000:.1f} ms  ({seq_ips:.1f} img/s)")
    print(f"batched/{args.batch} {args.n} images: {batched_s*1000:.1f} ms  ({bat_ips:.1f} img/s)")
    print(f"throughput speedup: {bat_ips / seq_ips:.2f}x")

    # p95 of single-image latency after warmup
    times = []
    for t in tensors[:64]:
        t0 = time.perf_counter()
        engine._run(t.unsqueeze(0))
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p95 = times[int(0.95 * (len(times) - 1))]
    print(f"single-image p95: {p95:.2f} ms")


if __name__ == "__main__":
    main()
