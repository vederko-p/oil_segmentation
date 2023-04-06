
from abc import ABC, abstractmethod
import numpy as np
from loguru import logger


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
        mask = (np.random.random((h, w)) > 0.5).astype(int)
        return mask
