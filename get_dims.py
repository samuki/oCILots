#!/usr/bin/env python

"""
Get information about images in a folder.
"""

from os import listdir
from os.path import isfile, join

from PIL import Image


def print_data(data):
    """
    Parameters
    ----------
    data : dict
    """
    for k, v in data.items():
        print("%s:\t%s" % (k, v))
    print("Min width: %i" % data["min_width"])
    print("Max width: %i" % data["max_width"])
    print("Min height: %i" % data["min_height"])
    print("Max height: %i" % data["max_height"])


def main(path):
    """
    Parameters
    ----------
    path : str
        Path where to look for image files.
    """
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # Filter files by extension
    onlyfiles = [f for f in onlyfiles if f.endswith(".png")]

    data = {}
    data["images_count"] = len(onlyfiles)
    data["min_width"] = 10 ** 100  # No image will be bigger than that
    data["max_width"] = 0
    data["min_height"] = 10 ** 100  # No image will be bigger than that
    data["max_height"] = 0

    for filename in onlyfiles:
        im = Image.open(path+filename)
        width, height = im.size
        if width not in [400, 1500] or height not in [400,1500]:
            print(filename, width, height)
        data["min_width"] = min(width, data["min_width"])
        data["max_width"] = max(width, data["max_width"])
        data["min_height"] = min(height, data["min_height"])
        data["max_height"] = max(height, data["max_height"])

    print_data(data)


if __name__ == "__main__":
    main(path="training/images/")
