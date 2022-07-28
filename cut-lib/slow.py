import time
import numpy as np
import networkx as nx  # type: ignore


# we use 2-tuples of ints as vertices, which also represent indices into the image
Vertex = tuple[int, int]


def edge_capacity(image: np.ndarray, u: Vertex, v: Vertex) -> float:
    return 1


def edge_capacity_s(image: np.ndarray, u: Vertex) -> float:
    return 1


def edge_capacity_t(image: np.ndarray, u: Vertex) -> float:
    return 1


def build_network(image: np.ndarray) -> nx.Graph:
    """Construct the s-t-network for the min-cut based on the given image."""
    # get image dimensions
    if len(image.shape) != 2:
        raise ValueError(
            f"build_network expects 2-dimensional image, but got {len(image.shape)} dimension"
        )
    w, h = image.shape
    # initialize graph and its nodes (source, sink, one node per pixel)
    G = nx.Graph()
    G.add_nodes_from(["s", "t"])
    G.add_nodes_from(((i, j) for i in range(w) for j in range(h)))
    # add edges from source to each pixel and from each pixel to sink
    for i in range(w):
        for j in range(h):
            G.add_edge("s", (i, j), capacity=edge_capacity_s(image, (i, j)))
            G.add_edge((i, j), "t", capacity=edge_capacity_t(image, (i, j)))
    # add edges from each (non right or bottom border) pixel to its bottom and right neighbor
    for i in range(w - 1):
        for j in range(h - 1):
            G.add_edge(
                (i, j), (i + 1, j), capacity=edge_capacity(image, (i, j), (i + 1, j))
            )
            G.add_edge(
                (i, j), (i, j + 1), capacity=edge_capacity(image, (i, j), (i, j + 1))
            )
    # add edges from right border pixels to bottom neighbor
    for j in range(h - 1):
        G.add_edge(
            (w - 1, j),
            (w - 1, j + 1),
            capacity=edge_capacity(image, (w - 1, j), (w - 1, j + 1)),
        )
    # add edges from bottom border pixels to right neighbor
    for i in range(w - 1):
        G.add_edge(
            (i, h - 1),
            (i + 1, h - 1),
            capacity=edge_capacity(image, (i, h - 1), (i + 1, h - 1)),
        )
    return G


def _min_cut_segmentation(image: np.ndarray) -> np.ndarray:
    """
    Compute a cut-based image segmentation based on the initial pixel weights given in the input.
    """
    # construct the corresponding flow-network
    start = time.perf_counter()
    G = build_network(image)
    end = time.perf_counter()
    print(f"Built graph in {end-start:.4f} seconds")
    # compute a minimum cut
    start = time.perf_counter()
    cut_value, (zeros, ones) = nx.minimum_cut(G, "s", "t", "capacity")
    end = time.perf_counter()
    print(f"Computed min-cut in {end-start:.4f} seconds")
    # set the pixel values according to the cut
    start = time.perf_counter()
    for u in zeros:
        if u == 's':
            continue
        i, j = u
        image[i, j] = 0
    for v in ones:
        if v == 't':
            continue
        i, j = v
        image[i, j] = 1
    end = time.perf_counter()
    print(f"Constructed result in {end-start:.4f} seconds")
    return image


def min_cut_segmentation(images: np.ndarray) -> np.ndarray:
    """
    Batched version of the above.
    """
    d = len(images.shape)
    if d == 2:
        return _min_cut_segmentation(images)
    elif d == 3:
        n, _, _ = images.shape
        for i in range(n):
            images[i, :, :] = _min_cut_segmentation(images[i, :, :])
        return images
    raise ValueError(
        f"min_cut_segmentation expects 2- or 3-dimensional input image(s), but got {d} dimensions"
    )


def benchmark(n: int):
    im = np.random.random((n, n))
    start = time.perf_counter()
    _ = _min_cut_segmentation(im)
    end = time.perf_counter()
    print(f"Finished {n}x{n} segmentation in {end-start:.4f} seconds")
