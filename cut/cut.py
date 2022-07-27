import ctypes
import numpy as np
from visualize import visualize

LIB = ctypes.CDLL("./include/cut-seg.so")


def to_ctypes(x: np.ndarray):
    return (
        x.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_uint(x.ndim),
        x.ctypes.shape_as(ctypes.c_uint),
        x.ctypes.strides_as(ctypes.c_uint),
    )


def rbf_log_segment(
    sigma: float, lambd: float, resolution: int, images: np.ndarray
) -> np.ndarray:
    segmentations = np.empty_like(images, dtype=np.int64)
    LIB.rbf_log_segment(
        ctypes.c_double(sigma),
        ctypes.c_double(lambd),
        ctypes.c_uint(resolution),
        ctypes.c_uint(images.ndim),
        images.ctypes.shape_as(ctypes.c_uint),
        images.ctypes.data_as(ctypes.c_void_p),
        images.ctypes.strides_as(ctypes.c_uint),
        segmentations.ctypes.data_as(ctypes.c_void_p),
        segmentations.ctypes.strides_as(ctypes.c_uint),
    )
    return segmentations


def main() -> None:
    images = np.load("../../data/predictions.npy")
    #  images = np.array(
    #      [
    #          [
    #              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #              [0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
    #              [0.1, 0.1, 0.9, 0.4, 0.1, 0.1],
    #              [0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
    #              [0.6, 0.1, 0.9, 0.1, 0.1, 0.1],
    #              [0.1, 0.1, 0.9, 0.9, 0.1, 0.1],
    #          ]
    #      ]
    #  )
    #  images = images[:, 52:56, 2:6]
    images = images[0, :, :].reshape((1, *images.shape[1:]))
    segmentations = {
        f"s{sigma}_l{lambd}_r{res}": rbf_log_segment(sigma, lambd, res, images)
        for sigma in (1.0,)
        for lambd in (1.0,)
        for res in (100,)
    }
    visualize(images, segmentations)


if __name__ == "__main__":
    main()
