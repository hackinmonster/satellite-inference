"""Build a TensorRT engine from the ONNX model. Needs NVIDIA GPU + TensorRT.

Falls back to a clear error if TensorRT isn't installed so the rest of the
project still runs on CPU via PyTorch.
"""

import argparse
import sys


def build(onnx_path, engine_path, fp16=True):
    try:
        import tensorrt as trt
    except ImportError:
        print("tensorrt python package not found")
        print("install NVIDIA TensorRT and rerun this on a machine with a GPU")
        sys.exit(0)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("fp16 enabled")
    else:
        print("building fp32 engine")

    # dynamic batch 1..16
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 64, 64), (8, 3, 64, 64), (16, 3, 64, 64))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("engine build failed")
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"wrote {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="weights/landcover.onnx")
    parser.add_argument("--out", default="weights/landcover.engine")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    build(args.onnx, args.out, fp16=not args.fp32)
