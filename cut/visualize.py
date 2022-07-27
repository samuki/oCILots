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

Images = NDArray[Shape["144, 400, 400"], Float]


def visualize(predictions: Images, segmentations: dict[str, Images]) -> None:
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
    # add colorbar
    fig.colorbar(round_image)

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

    fig.show()
    plt.show()
