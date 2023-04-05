
import torch
import matplotlib.pyplot as plt

import feature_extractor as f_extr


def show_img_lbl(img_p, lbl_p):
    image, _, binary_mask = f_extr.parse_img_label(img_p, lbl_p)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image)
    ax[0].set_title('Source Image', fontsize=15)

    ax[1].imshow(binary_mask, 'gray')
    ax[1].set_title('Segmentation Mask', fontsize=15)

    plt.show()


def plot_ds_tens(img_tens, lbl_tens):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img_tens.numpy().transpose(1, 2, 0))
    ax[0].set_title('Source Image', fontsize=15)

    ax[1].imshow(lbl_tens.squeeze().numpy(), 'gray')
    ax[1].set_title('Segmentation Mask', fontsize=15)

    plt.show()


def plot_examples_predict(model, dataset, images_ids, save_name=None):
    rows = len(images_ids)
    cols = 3

    fig, ax = plt.subplots(rows, cols, figsize=(12, rows * 3))
    ax[0, 0].set_title('Source image')
    ax[0, 1].set_title('True mask')
    ax[0, 2].set_title('Predict mask')

    for i, img_id in enumerate(images_ids):
        tens_img, tens_mask = dataset[img_id]
        model_input = tens_img.unsqueeze(0).to('cuda')
        with torch.no_grad():
            pred_mask = (model(model_input) > 0.5).float()

        ax[i, 0].imshow(tens_img.numpy().transpose(1, 2, 0))
        ax[i, 1].imshow(tens_mask.numpy().transpose(1, 2, 0), 'gray')
        ax[i, 2].imshow(pred_mask.cpu().squeeze().numpy(), 'gray')

    if save_name is not None:
        plt.savefig(save_name)

    plt.show()
