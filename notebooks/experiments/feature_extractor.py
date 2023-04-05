
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def read_masks(filepath, img_h, img_w):
    # read label file:
    with open(filepath, 'r') as lbl_file:
        lines = lbl_file.readlines()
    # label lines --> list of points:
    all_points = [
        [float(x) for x in line[2:].split(' ')]
        for line in lines
    ]
    # list of [0,1] points --> array of [0, img_size] points:
    poly_masks = [
        np.array(shape).reshape(-1, 2) * np.array([[img_h, img_w]])
        for shape in all_points
    ]
    # make binary mask:
    color = (1, 1, 1)
    binary_mask = np.zeros((img_h, img_w))
    cv2.fillPoly(binary_mask, [m.astype(int) for m in poly_masks], color)
    return poly_masks, binary_mask.astype(int)


def parse_img_label(img_p, lbl_p):
    image = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
    img_h, img_w, img_ch = image.shape
    if lbl_p.split('.')[-1] == 'txt':
        poly_masks, binary_mask = read_masks(lbl_p, img_h, img_w)
    else:
        poly_masks = None
        binary_mask = cv2.cvtColor(cv2.imread(lbl_p), cv2.COLOR_BGR2GRAY)
        if binary_mask.max() > 1:
            binary_mask = (binary_mask / 255).round().astype(int)
    return image, poly_masks, binary_mask


def vis_label(img_p, lbl_p):
    # read image, pnt masks, binary mask:
    image, poly_masks, binary_mask = parse_img_label(img_p, lbl_p)
    # draw
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    # Source image:
    ax[0].imshow(image)
    ax[0].set_title('Source Image')
    # Poly mask:
    if poly_masks is not None:
        patches = PatchCollection(
            [Polygon(mask) for mask in poly_masks],
            alpha=0.4, color='orange')
        ax[1].imshow(image)
        ax[1].add_collection(patches)
    else:
        ax[1].imshow(image)
    ax[1].set_title('Poly mask')
    # Binary mask:
    ax[2].imshow(binary_mask*255, 'gray')
    ax[2].set_title('Binary mask')

    for ax_i in ax:
        ax_i.axis('off')

    plt.show()


def get_pixelwise_features(img_p, lbl_p):
    """Extracts pixels from image. Returns df."""
    # read image, pnt masks, binary mask:
    image, masks, binary_mask = parse_img_label(img_p, lbl_p)
    img_h, img_w, img_ch = image.shape
    # pixelwise transform iamge to table:
    # would be helpful for mean of area:
    # ~ np.lib.stride_tricks.sliding_window_view(
    #     t, (1, 1), axis=(0, 1)
    # ).reshape(img_h*img_w, img_ch)
    df = pd.DataFrame(
        image.reshape(img_h * img_w, img_ch),
        columns=['R', 'G', 'B']
    )
    df['spill'] = binary_mask.flatten().astype(int)
    return df
