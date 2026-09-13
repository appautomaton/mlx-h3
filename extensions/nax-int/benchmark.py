import statistics
import time

import mlx.core as mx

from mlx_nax_int import matmul


M = 2048
N = 2048
K = 4096
WARMUPS = 5
ITERATIONS = 30
TRIALS = 3


def measure(name, operation):
    for _ in range(WARMUPS):
        mx.async_eval(operation())
    mx.synchronize()

    elapsed = []
    for _ in range(TRIALS):
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            mx.async_eval(operation())
        mx.synchronize()
        elapsed.append(time.perf_counter() - start)

    seconds = statistics.median(elapsed)
    milliseconds = seconds * 1000 / ITERATIONS
    operations = 2 * M * N * K * ITERATIONS
    teraops = operations / seconds / 1e12
    print(f"{name:<8} {milliseconds:7.3f} ms  {teraops:7.2f} TOp/s")


if __name__ == "__main__":
    fp16_x = mx.zeros((M, K), dtype=mx.float16)
    fp16_weight = mx.zeros((K, N), dtype=mx.float16)
    int8_x = mx.zeros((M, K), dtype=mx.int8)
    int8_weight = mx.zeros((K, N), dtype=mx.int8)
    int4_weight = mx.zeros((K, N // 2), dtype=mx.uint8)
    mx.eval(fp16_x, fp16_weight, int8_x, int8_weight, int4_weight)

    print(f"M={M} N={N} K={K}; median of {TRIALS} x {ITERATIONS} runs")
    measure("FP16", lambda: mx.matmul(fp16_x, fp16_weight))
    measure("W8A8", lambda: matmul(int8_x, int8_weight, 8))
    measure("W4A8", lambda: matmul(int8_x, int4_weight, 4))
