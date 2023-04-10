
import os
from loguru import logger

import cv2
import numpy as np

from src.infrastructure.read_video import read_video
from src.infrastructure.segmentation import SegmentationModel
from src.infrastructure.image_slicing import crop_image_by_edges


class SegmentationService:

    def __init__(self, segmentation_model: SegmentationModel):
        self.segmentation_model = segmentation_model

    def _process_video(self, video_path: str) -> np.ndarray:
        logger.info('Processing video')
        video_frames = read_video(video_path)
        os.remove(video_path)
        logger.info('Stitching frames')
        stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        status, output = stitcher.stitch(video_frames)
        if status != 0:
            logger.warning('Cant stitch frames into single image')
            # TODO: Handle stitching exception
        output = crop_image_by_edges(output).copy()
        return output

    def _draw_mask(self, image: np.ndarray, mask: np.ndarray):
        mask = np.array(mask).sum(axis=0).astype(bool)
        h, w, c = image.shape
        res = (
                np.stack([mask for _ in range(c)]).transpose(1, 2, 0)
                * np.ones((h, w, c)) * np.array([87, 87, 0])
        ).astype(int)
        res += image
        res = np.where(res > 255, 255, res) * (image > 0)
        return res

    def recognize_oil_spill(self, video_path: str):
        stitched_image = self._process_video(video_path)
        logger.info('Oil Spills segmentation process')
        oil_spill_mask = self.segmentation_model.segment(stitched_image)
        stitched_image_with_mask = self._draw_mask(
            stitched_image, oil_spill_mask
        )
        return stitched_image, stitched_image_with_mask
