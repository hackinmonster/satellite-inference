import asyncio
import time

from preprocess import load_image


class Batcher:
    def __init__(self, engine, max_batch=16, max_wait_ms=50):
        self.engine = engine
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self._worker())

    async def infer(self, image_bytes: bytes):
        tensor = await asyncio.to_thread(load_image, image_bytes)
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((tensor, fut, time.perf_counter()))
        return await fut

    async def _worker(self):
        while True:
            tensor, fut, t0 = await self.queue.get()
            batch = [(tensor, fut, t0)]
            deadline = time.perf_counter() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            tensors = [item[0] for item in batch]
            try:
                results, _ = await asyncio.to_thread(self.engine.predict_many, tensors)
            except Exception as e:
                for _, f, _ in batch:
                    if not f.done():
                        f.set_exception(e)
                continue

            now = time.perf_counter()
            for result, (_, f, started) in zip(results, batch):
                result["latency_ms"] = round((now - started) * 1000, 2)
                result["backend"] = self.engine.backend
                result["batch_size"] = len(batch)
                if not f.done():
                    f.set_result(result)
