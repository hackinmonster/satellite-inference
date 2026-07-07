import numpy as np


class TRTEngine:
    """Thin TensorRT wrapper. Import-heavy on purpose so Engine can catch
    a missing GPU stack and fall back to PyTorch."""

    def __init__(self, path):
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda

        self.trt = trt
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def predict(self, batch):
        """batch: float32 numpy NCHW. returns logits N x C"""
        n = batch.shape[0]
        self.context.set_input_shape("images", batch.shape)

        d_in = self.cuda.mem_alloc(batch.nbytes)
        out_shape = (n, self.engine.get_tensor_shape("logits")[-1])
        h_out = np.empty(out_shape, dtype=np.float32)
        d_out = self.cuda.mem_alloc(h_out.nbytes)

        self.cuda.memcpy_htod_async(d_in, np.ascontiguousarray(batch), self.stream)
        self.context.set_tensor_address("images", int(d_in))
        self.context.set_tensor_address("logits", int(d_out))
        self.context.execute_async_v3(self.stream.handle)
        self.cuda.memcpy_dtoh_async(h_out, d_out, self.stream)
        self.stream.synchronize()
        d_in.free()
        d_out.free()
        return h_out
