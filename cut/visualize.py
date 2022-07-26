from typing import cast
import os
import time
import itertools
import math
import numpy as np
from nptyping import NDArray, Float, Shape
import matplotlib.pyplot as plt  # type: ignore

W = 400
H = 400
N = W * H
N_IMAGES = 144

PREDICTIONS_FILENAME_NPY = "../../data/predictions.npy"
PREDICTIONS_FILENAME_TXT = "../../data/predictions_flattened.csv"
SEGMENTATIONS_DIR = "../data/segmentations"

Images = NDArray[Shape["144, 400, 400"], Float]


def load_predictions(method: str) -> Images:
    # load and reshape predictions & segmentations
    start = time.perf_counter()
    if method == "npy":
        predictions = np.load(PREDICTIONS_FILENAME_NPY)
    elif method == "csv":
        predictions = np.loadtxt(PREDICTIONS_FILENAME_TXT)
        predictions = predictions.reshape((N_IMAGES, W, H))
    else:
        raise ValueError(f"method must be 'npy' or 'csv', but was '{method}'")
    end = time.perf_counter()
    print(f"loaded predictions in {end-start:.4f} seconds")
    return cast(Images, predictions)


def load_segmentations(dir_name: str) -> dict[str, Images]:
    start = time.perf_counter()
    segmentations = {}
    fnames = os.listdir(dir_name)
    n_segs = len(fnames)
    for i, filename in enumerate(fnames):
        print(f"\rloading segmentation {i+1:{len(str(n_segs))}d}/{n_segs} ... ", end="")
        segs = np.loadtxt(f"{dir_name}/{filename}")
        segs = segs.reshape((N_IMAGES, W, H))
        segmentations[filename] = segs
    end = time.perf_counter()
    print(
        f"\rloaded {n_segs} segmentations in {end-start:.4f} seconds "
        f"({(end-start)/n_segs:.4f} per segmentation)"
    )
    return segmentations


def main() -> None:
    # load prediction & segmentation images
    predictions = load_predictions('npy')
    segmentations = load_segmentations(SEGMENTATIONS_DIR)
    n_segmentations = len(segmentations)
    n_subplots = n_segmentations + 2
    n_rows = int(math.ceil(math.sqrt(n_subplots)))

    curr_image = 0

    # plot prediction heatmap and segmentation side-by-side
    fig, axs = plt.subplots(n_rows, n_rows, figsize=(16, 10), constrained_layout=True)
    pred_ax, round_ax = axs.flat[0], axs.flat[1]
    pred_ax.title.set_text("prediction")
    round_ax.title.set_text("rounded")
    pred_image = pred_ax.imshow(predictions[curr_image, :, :])
    round_image = round_ax.imshow(np.round(predictions[curr_image, :, :]))
    cut_images = []
    for i, (ax, (segmenter, segs)) in enumerate(
        zip(itertools.islice(axs.flat, 2, None), segmentations.items())
    ):
        ax.title.set_text(segmenter)
        cut_images.append(ax.imshow(segs[curr_image, :, :]))
    #  fig.subplot_tool()

    def show_image() -> None:
        nonlocal curr_image
        pred_image.set_data(predictions[curr_image, :, :])
        round_image.set_data(np.round(predictions[curr_image, :, :]))
        for im, segs in zip(cut_images, segmentations.values()):
            im.set_data(segs[curr_image, :, :])
        fig.canvas.draw()

    def left_callback() -> None:
        nonlocal curr_image
        curr_image -= 1
        show_image()

    def right_callback() -> None:
        nonlocal curr_image
        curr_image += 1
        show_image()

    toolbar_elements = fig.canvas.toolbar.children()
    left_button, right_button = toolbar_elements[6], toolbar_elements[8]
    left_button.setEnabled(True)
    right_button.setEnabled(True)
    left_button.clicked.connect(left_callback)
    right_button.clicked.connect(right_callback)

    fig.show()
    plt.show()


if __name__ == "__main__":
    main()
