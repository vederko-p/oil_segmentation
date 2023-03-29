
import numpy as np


def masks_tensors_iou(tens_0, tens_1):
    intersection = (tens_0 * tens_1).sum(axis=(1, 2, 3))
    union = ((tens_0 + tens_1) > 0).sum(axis=(1, 2, 3))
    return intersection / union


def masks_iou(mask1: np.array, mask2: np.array):
    iou = (mask1 * mask2).sum() / ((mask1 + mask2) > 0).sum()
    return round(iou, 3)
