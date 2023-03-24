
from math import ceil
import numpy as np


def crop_image_by_edges(img):
    mask_y, mask_x = np.where(img.sum(axis=2) > 0)
    return img[
        mask_y.min(): mask_y.max(),
        mask_x.min(): mask_x.max()
    ]


def padding_before_sub(img, w_size):
    """Padding image to fit whole image in same slices."""

    # Calculate padding:
    source_shape = img.shape[:-1]
    target_shape = tuple([
        ceil(x / w_size) * w_size
        for x in img.shape[:-1]
    ])
    diff_shape = tuple([
        ts - ss for ss, ts in zip(source_shape, target_shape)
    ])
    padding_shape = tuple(
        [(s // 2, s - (s // 2)) for s in diff_shape]
    )

    # Padding:
    return np.stack([
        np.pad(img[:, :, ch], padding_shape)
        for ch in range(img.shape[-1])
    ]).transpose(1, 2, 0)
