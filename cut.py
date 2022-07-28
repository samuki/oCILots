from typing import Optional, Any, Callable
import ctypes
import sys
import time
import math
import itertools
import glob
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import config
import dataset
import utils


LIB = ctypes.CDLL("./cut-lib/include/cut-seg.so")


def plot(pred: np.ndarray, gt: np.ndarray, seg: np.ndarray):
    cmap_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}
    fig, axs = plt.subplots(1, 3, figsize=(16, 10), constrained_layout=True)
    pred_ax, gt_ax, seg_ax = axs.flat
    pred_ax.title.set_text("prediction")
    gt_ax.title.set_text("ground-truth")
    seg_ax.title.set_text("segmentation")
    pred_ax.imshow(pred, **cmap_args)
    gt_ax.imshow(gt, **cmap_args)
    seg_ax.imshow(seg, **cmap_args)
    fig.show()
    plt.show()


def rbf_log_segment(
    images: np.ndarray, *, sigma: float, lambd: float, resolution: int
) -> np.ndarray:
    segmentations = np.empty_like(images, dtype=np.int32)
    sigma_ct: ctypes.c_float | ctypes.c_double
    lambd_ct: ctypes.c_float | ctypes.c_double
    if images.dtype == np.float32:
        fn = LIB.rbf_log_segment_float
        sigma_ct = ctypes.c_float(sigma)
        lambd_ct = ctypes.c_float(lambd)
    elif images.dtype == np.float64:
        fn = LIB.rbf_log_segment_double
        sigma_ct = ctypes.c_double(sigma)
        lambd_ct = ctypes.c_double(lambd)
    else:
        raise ValueError(f"images have unknown datatype {images.dtype!r}")
    fn(
        sigma_ct,
        lambd_ct,
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


def load_prediction_groundtruth(base_dir: str) -> tuple[np.ndarray, np.ndarray]:
    n_images = len(glob.glob(f"{base_dir}/pred_*.npy"))
    pred = np.load(f"{base_dir}/pred_0.npy")
    gt = np.load(f"{base_dir}/gt_0.npy")
    w, h = pred.shape
    preds, gts = np.empty((n_images, w, h)), np.empty((n_images, w, h), dtype=np.int32)
    preds[0, :, :], gts[0, :, :] = pred, gt
    for i in range(1, n_images):
        preds[i, :, :] = np.load(f"{base_dir}/pred_{i}.npy")
        gts[i, :, :] = np.load(f"{base_dir}/gt_{i}.npy")
    return preds, gts


def validate_segmenters(
    segmenters: list[tuple[str, dict[str, Any]]],
    predictions: np.ndarray,
    groundtruths: np.ndarray,
    metric_fns: dict[str, Callable[[Any, Any], np.double]],
    *,
    rank_metric: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    total_start = time.perf_counter()
    groundtruths_tensor = torch.tensor(groundtruths.astype('float32'))
    # run and validate all segmenters on all metrics
    n_segmenters = len(segmenters)
    metrics = {name: np.empty((n_segmenters,)) for name in metric_fns}
    metric_width = max(len(name) for name in metric_fns)
    for i, (method, args) in enumerate(segmenters):
        print(
            f"{i+1:{len(str(n_segmenters))}d}/{n_segmenters}: {method}{args} ... ",
            end="",
        )
        sys.stdout.flush()
        start = time.perf_counter()
        segmentations = segment(predictions, method, **args).astype("float32")
        segmentations_tensor = torch.tensor(segmentations)
        end = time.perf_counter()
        print(f"done in {end-start:.4f} seconds")
        print(f"differ in {np.count_nonzero(segmentations[0]-groundtruths[0])} elements "
              f"with max-diff {np.max(np.abs(segmentations[0]-groundtruths[0]))}")
        for name, fn in metric_fns.items():
            metric = fn(groundtruths_tensor, segmentations_tensor)
            metrics[name][i] = metric
            print(f"\t{name:{metric_width}s}  {metric}")
    # extract and print the best-performing one if a rank-metric is given
    if rank_metric is not None:
        best_idx = np.argmax(metrics[rank_metric])
        best = segmenters[best_idx]
        print(f"Best: {best[0]}{best[1]} with {metrics[rank_metric][best_idx]}")
    end = time.perf_counter()
    print(f"Total time: {end-total_start:.4f} seconds")
    return best


def main() -> None:
    predictions, groundtruths = load_prediction_groundtruth(
        "results/28072022_15:48:50/training_predictions"
    )
    segmenters = [
        ("rbf-log", {"sigma": sigma, "lambd": lambd, "resolution": res})
        for sigma in (0.1, 1.0, 10.0)
        for lambd in np.arange(start=0.1, stop=1.0, step=0.02)
        for res in (100, 1000)
    ]
    validate_segmenters(
        segmenters,
        predictions,
        groundtruths,
        config.METRICS,
        rank_metric="patch_f1_fn",
    )


if __name__ == "__main__":
    main()