
import numpy as np


def masks_iou(mask1: np.array, mask2: np.array):
    iou = (mask1 == mask2).sum() / ((mask1 + mask2) > 0).sum()
    return round(iou, 3)
