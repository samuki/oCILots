import ctypes
import sys
import time
import numpy as np
from visualize import visualize

LIB = ctypes.CDLL("./include/cut-seg.so")


def rbf_log_segment(
    images: np.ndarray, *, sigma: float, lambd: float, resolution: int
) -> np.ndarray:
    segmentations = np.empty_like(images, dtype=np.int32)
    LIB.rbf_log_segment(
        ctypes.c_float(sigma),
        ctypes.c_float(lambd),
        ctypes.c_uint(resolution),
        ctypes.c_uint(images.ndim),
        images.ctypes.shape_as(ctypes.c_uint),
        images.ctypes.data_as(ctypes.c_void_p),
        images.ctypes.strides_as(ctypes.c_uint),
        segmentations.ctypes.data_as(ctypes.c_void_p),
        segmentations.ctypes.strides_as(ctypes.c_uint),
    )
    return segmentations


def segment(images: np.ndarray, method: str, *, benchmark: bool = False, **kwargs):
    if benchmark:
        start = time.perf_counter()
        print(f"segmenting with {method}{kwargs} ... ", end="")
        sys.stdout.flush()
    if method == "rbf-log":
        seg = rbf_log_segment(images, **kwargs)
    else:
        raise ValueError(f"unknown method '{method}'")
    if benchmark:
        end = time.perf_counter()
        print(f"done in {end-start:.4f} seconds")
    return seg


def main() -> None:
    images = np.load("../../data/predictions.npy")
    #  images = images[0, :, :].reshape((1, *images.shape[1:]))
    segmentations = {
        f"s{sigma}_l{lambd}_r{res}": segment(
            images, "rbf-log", sigma=sigma, lambd=lambd, resolution=res, benchmark=True
        )
        for sigma in (0.1, 1.0, 10.0)
        for lambd in (0.1, 1.0, 10.0)
        for res in (100,)
    }
    visualize(images, segmentations, show=True)


if __name__ == "__main__":
    main()
