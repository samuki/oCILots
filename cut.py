from typing import Optional
import ctypes
import os
import sys
import time
import math
import itertools
from datetime import datetime
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import config
import dataset
import utils


LIB = ctypes.CDLL("./cut-lib/include/cut-seg.so")


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


def visualize(
    predictions: np.ndarray,
    segmentations: dict[str, np.ndarray],
    *,
    show: bool = True,
    save: Optional[str] = None,
) -> None:
    # load prediction & segmentation images
    n_segmentations = len(segmentations)
    n_subplots = n_segmentations + 2
    n_rows = int(math.ceil(math.sqrt(n_subplots)))

    n_images = predictions.shape[0]
    curr_image = 0

    cmap_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}

    # plot prediction heatmap and segmentation side-by-side
    fig, axs = plt.subplots(n_rows, n_rows, figsize=(16, 10), constrained_layout=True)
    pred_ax, round_ax = axs.flat[0], axs.flat[1]
    pred_ax.title.set_text("prediction")
    round_ax.title.set_text("rounded")
    pred_image = pred_ax.imshow(predictions[curr_image, :, :], **cmap_args)
    round_image = round_ax.imshow(np.round(predictions[curr_image, :, :]), **cmap_args)
    cut_images = []
    for i, (ax, (segmenter, segs)) in enumerate(
        zip(itertools.islice(axs.flat, 2, None), segmentations.items())
    ):
        ax.title.set_text(segmenter)
        cut_images.append(ax.imshow(segs[curr_image, :, :], **cmap_args))

    def show_image() -> None:
        pred_image.set_data(predictions[curr_image, :, :])
        round_image.set_data(np.round(predictions[curr_image, :, :]))
        for im, segs in zip(cut_images, segmentations.values()):
            im.set_data(segs[curr_image, :, :])
        fig.canvas.draw()

    def left_callback() -> None:
        nonlocal curr_image
        if curr_image == 0:
            return
        curr_image -= 1
        show_image()

    def right_callback() -> None:
        nonlocal curr_image
        if curr_image == n_images - 1:
            return
        curr_image += 1
        show_image()

    toolbar_elements = fig.canvas.toolbar.children()
    left_button, right_button = toolbar_elements[6], toolbar_elements[8]
    left_button.setEnabled(True)
    right_button.setEnabled(True)
    left_button.clicked.connect(left_callback)
    right_button.clicked.connect(right_callback)

    if show:
        fig.show()
        plt.show()
    if save is not None:
        fig.savefig(save)


def make_submission(test_pred: np.ndarray, cutoff: float, filename: str) -> None:
    _, size, test_filenames = dataset.load_test_data()
    test_pred = test_pred.reshape(
        (
            -1,
            size[0] // config.PATCH_SIZE,
            config.PATCH_SIZE,
            size[1] // config.PATCH_SIZE,
            config.PATCH_SIZE,
        )
    )
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > cutoff)
    utils.create_submission(test_pred, test_filenames, filename)


def main() -> None:
    images = np.load("data/predictions.npy")
    #  images = images[0, :, :].reshape((1, *images.shape[1:]))
    segmentations = {
        f"s{sigma}_l{lambd}_r{res}": segment(
            images, "rbf-log", sigma=sigma, lambd=lambd, resolution=res, benchmark=True
        )
        for sigma in (1.0,)
        #  for lambd in (0.4, 0.45, 0.5, 0.55)
        for lambd in (0.45,)
        for res in (100,)
    }
    visualize(images, segmentations, show=False)
    dt_string = datetime.now().strftime("results/%d%m%Y_%H:%M:%S")
    submission_dir = f"results/{dt_string}"
    os.makedirs(submission_dir)
    submission_filename = f"{submission_dir}/cut-segmentation.csv"
    make_submission(
        test_pred=segmentations["s1.0_l0.45_r100"],
        cutoff=config.CUTOFF,
        filename=submission_filename,
    )


if __name__ == "__main__":
    main()
