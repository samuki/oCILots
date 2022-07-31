from typing import Optional, Any, Callable, Sequence
import abc
import ctypes
import sys
import time
import math
import glob
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import config
import dataset
import utils


LIB = ctypes.CDLL("./cut-lib/include/cut-seg.so")


class Segmenter(abc.ABC):
    resolution: int

    def __init__(self, resolution: int) -> None:
        self.resolution = resolution

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractclassmethod
    def segment(self, images: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, images: np.ndarray) -> np.ndarray:
        return self.segment(images)


class RBFLogSegmenter(Segmenter):
    sigma: float
    lambd: float

    def __init__(self, sigma: float, lambd: float, resolution: int) -> None:
        super().__init__(resolution)
        self.sigma = sigma
        self.lambd = lambd

    def __str__(self) -> str:
        return f"RBFLog(sigma={self.sigma}, lambda={self.lambd}, res={self.resolution})"

    def segment(self, images: np.ndarray) -> np.ndarray:  # type: ignore
        segmentations = np.empty_like(images, dtype=np.int32)
        sigma_ct: ctypes.c_float | ctypes.c_double
        lambd_ct: ctypes.c_float | ctypes.c_double
        if images.dtype == np.float32:
            fn = LIB.rbf_log_segment_float
            sigma_ct = ctypes.c_float(self.sigma)
            lambd_ct = ctypes.c_float(self.lambd)
        elif images.dtype == np.float64:
            fn = LIB.rbf_log_segment_double
            sigma_ct = ctypes.c_double(self.sigma)
            lambd_ct = ctypes.c_double(self.lambd)
        else:
            raise ValueError(f"images have unknown datatype {images.dtype!r}")
        fn(
            sigma_ct,
            lambd_ct,
            ctypes.c_uint(self.resolution),
            ctypes.c_uint(images.ndim),
            images.ctypes.shape_as(ctypes.c_uint),
            images.ctypes.data_as(ctypes.c_void_p),
            images.ctypes.strides_as(ctypes.c_uint),
            segmentations.ctypes.data_as(ctypes.c_void_p),
            segmentations.ctypes.strides_as(ctypes.c_uint),
        )
        return segmentations


class DirectionSegmenter(Segmenter):
    lambda_pred: int
    lambda_dir: int
    radsiu: int
    delta_theta: float

    def __init__(
        self,
        lambda_pred: int,
        lambda_dir: int,
        radius: int,
        delta_theta: float,
        resolution: int,
    ) -> None:
        super().__init__(resolution)
        self.lambda_pred = lambda_pred
        self.lambda_dir = lambda_dir
        self.radius = radius
        self.delta_theta = delta_theta

    def __str__(self) -> str:
        return (
            f"Direction(pred={self.lambda_pred}, dir={self.lambda_dir}, r={self.radius}, "
            f"dt={self.delta_theta:.4f})"
        )

    def segment(self, images: np.ndarray) -> np.ndarray:  # type: ignore
        segmentations = np.empty_like(images, dtype=np.int32)
        if images.dtype != np.int32:
            raise ValueError(f"images have unknown datatype {images.dtype!r}")
        LIB.dir_segment_unsigned(
            ctypes.c_uint(self.lambda_pred),
            ctypes.c_uint(self.lambda_dir),
            ctypes.c_uint(self.radius),
            ctypes.c_double(self.delta_theta),
            ctypes.c_uint(self.resolution),
            ctypes.c_uint(images.ndim),
            images.ctypes.shape_as(ctypes.c_uint),
            images.ctypes.data_as(ctypes.c_void_p),
            images.ctypes.strides_as(ctypes.c_uint),
            segmentations.ctypes.data_as(ctypes.c_void_p),
            segmentations.ctypes.strides_as(ctypes.c_uint),
        )
        return segmentations


def visualize(
    *,
    show: bool = True,
    save: Optional[str] = None,
    **kwargs: np.ndarray | tuple[np.ndarray, dict[str, Any]],
) -> None:
    # load prediction & segmentation images
    n_rows = int(math.ceil(math.sqrt(len(kwargs))))

    fst = next(iter(kwargs.values()))
    if isinstance(fst, tuple):
        fst, _ = fst
    n_images = fst.shape[0]
    curr_image = 0

    default_cmap_args = {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}

    # plot prediction heatmap and segmentation side-by-side
    fig, axs = plt.subplots(n_rows, n_rows, figsize=(16, 10), constrained_layout=True)
    images = {}
    for (name, image), ax in zip(kwargs.items(), axs.flat):
        if isinstance(image, tuple):
            image, cmap_args = image
        else:
            cmap_args = default_cmap_args
        ax.title.set_text(name)
        images[name] = ax.imshow(image[curr_image, :, :], **cmap_args)
        plt.colorbar(images[name], ax=ax)

    def show_image() -> None:
        for name, im in images.items():
            image = kwargs[name]
            if isinstance(image, tuple):
                image, _ = image
            im.set_data(image[curr_image, :, :])
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
    segmenters: Sequence[Segmenter],
    predictions: np.ndarray,
    groundtruths: np.ndarray,
    metric_fns: dict[str, Callable[[Any, Any], np.double]],
    *,
    rank_metric: Optional[str] = None,
    print_sorted: bool = False,
) -> Segmenter:
    total_start = time.perf_counter()
    groundtruths_tensor = torch.tensor(groundtruths.astype("float32"))
    # run and validate all segmenters on all metrics
    n_segmenters = len(segmenters)
    metrics = {name: np.empty((n_segmenters,)) for name in metric_fns}
    metric_width = max(len(name) for name in metric_fns)
    for i, segmenter in enumerate(segmenters):
        print(
            f"{i+1:{len(str(n_segmenters))}d}/{n_segmenters}: {segmenter!s} ... ",
            end="\n",
        )
        sys.stdout.flush()
        start = time.perf_counter()
        segmentations = segmenter(predictions).astype("float32")
        segmentations_tensor = torch.tensor(segmentations)
        end = time.perf_counter()
        print(f"done in {end-start:.4f} seconds")
        for name, fn in metric_fns.items():
            metric = fn(groundtruths_tensor, segmentations_tensor)
            metrics[name][i] = metric
            print(f"\t{name:{metric_width}s}  {metric}")
    # extract and print the best-performing one if a rank-metric is given
    if rank_metric is not None:
        if not print_sorted:
            best_idx = int(np.argmax(metrics[rank_metric]))
            best = segmenters[best_idx]
            print(f"Best: {best!s} with {metrics[rank_metric][best_idx]}")
        else:
            segmenters_metrics = sorted(
                zip(segmenters, metrics[rank_metric]), key=lambda t: t[1]
            )
            for i, (seg, metric) in enumerate(segmenters_metrics):
                print(
                    f"{n_segmenters-i:{len(str(n_segmenters))}d}/{n_segmenters}: "
                    f"{seg!s} with {metric}"
                )
            best, _ = segmenters_metrics[-1]
    end = time.perf_counter()
    print(f"Total time: {end-total_start:.4f} seconds")
    return best


def main() -> None:
    base_dir = "results/predictions"
    predictions = np.load(f"{base_dir}/predictions.npy")
    groundtruths = np.load(f"{base_dir}/groundtruths.npy")
    segmenters = [
        RBFLogSegmenter(sigma=0.1, lambd=lambd, resolution=100)
        for lambd in np.arange(start=0.1, stop=0.7, step=0.01)
    ]
    validate_segmenters(
        segmenters,
        predictions,
        groundtruths,
        {**config.METRICS, "f1": utils.f1_fn},
        rank_metric="f1",
        print_sorted=True,
    )


if __name__ == "__main__":
    main()
