
from abc import ABC, abstractmethod
import numpy as np
import torch
import os
from loguru import logger
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import mmcv
import matplotlib.pyplot as plt


class SegmentationModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return binary segmentation mask"""
        pass


class DummySegmentation(SegmentationModel):
    def __init__(self, weights_path: str):
        super().__init__()
        logger.info('Loading Dummy Segmentation model')

    def segment(self, image):
        h, w, c = image.shape
        mask = np.zeros((h, w))
        mask[:h // 2, :w // 2] = 1
        return mask


class MMSegmentation(SegmentationModel):
    def __init__(self, weights_path: str, config_file: str):
        super().__init__()
        logger.info('Loading MM Segmentation model')
        self.cfg = mmcv.Config.fromfile(config_file)
        self.cfg.model.pretrained = True
        self.cfg.data.test.test_mode = True
        self.weights_path = weights_path
        if os.path.exists(config_file) and os.path.exists(weights_path):
            device = self.set_device()
            self.init_model(device = device)

    def set_device(self):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def init_model(self, device = 'cuda:0'):
        self.model = init_segmentor(self.cfg, self.weights_path, device=device)

    def segment(self, image):
        return inference_segmentor(self.model, image)

    def show_final(self, img: np.ndarray,
                       mask: np.ndarray,
                       out_path: str = '../static/output2.png'):
        plt.show(block = True)
        show_result_pyplot(self.model, img, mask, self.model.PALETTE, out_file = out_path)


if __name__ == '__main__':
    segmentator = MMSegmentation('../weights/pspnet.pth', '../configs/oil_seg_config.py')
    img = mmcv.imread('../static/input.png')
    mask = segmentator.segment(img)
    res_img = segmentator.show_final(img, mask)
