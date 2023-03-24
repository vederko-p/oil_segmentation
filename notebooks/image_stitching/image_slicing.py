
from math import ceil
import numpy as np


def crop_image_by_edges(img):
    mask_y, mask_x = np.where(img.sum(axis=2) > 0)
    return img[
        mask_y.min(): mask_y.max(),
        mask_x.min(): mask_x.max()
    ]
