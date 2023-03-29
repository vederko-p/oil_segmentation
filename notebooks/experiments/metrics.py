
import numpy as np


def masks_tensors_iou(tens_0, tens_1):
    intersection = (tens_0 * tens_1).sum(axis=(1, 2, 3))
    union = ((tens_0 + tens_1) > 0).sum(axis=(1, 2, 3))
    return intersection / union


def masks_iou(mask1: np.array, mask2: np.array):
    iou = (mask1 * mask2).sum() / ((mask1 + mask2) > 0).sum()
    return round(iou, 3)


def eval_iou_over_dataset(data_loader, model, p=0.5, device=None):
    """Будем считать средневзвешенное IoU по изображениям."""

    if device is None:
        _device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(_device_name)

    model.eval()

    mean_iou = 0
    total = 0
    for batch_x, batch_y in tqdm(data_loader):
        with torch.no_grad():
            batch_pred_probs = model(batch_x.to(device))
            batch_pred = (batch_pred_probs > p).int()
        batch_iou = masks_tensors_iou(batch_pred, batch_y.to(device)).mean()
        mean_iou += batch_iou * batch_x.shape[0]
        total += batch_x.shape[0]
    return mean_iou / total