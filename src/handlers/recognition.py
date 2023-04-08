
import os
from tempfile import NamedTemporaryFile

import cv2
from loguru import logger

from src.service.recognition import SegmentationService


class RecognitionHandler:

    def __init__(
            self, segmentation_service: SegmentationService,
            output_dir_path: str
    ):
        self.service = segmentation_service
        self.output_dir_path = output_dir_path

    def _save_temp_file(self, file):
        logger.info('Saving temporary file')
        temp = NamedTemporaryFile(delete=False)
        contents = file.file.read()
        with temp as f:
            f.write(contents)
        del contents
        return temp.name

    def _save_recognition_results(self, image, image_with_mask) -> None:
        logger.info('Writing segmentation results')
        input_filepath = os.path.join(self.output_dir_path, 'input.png')
        cv2.imwrite(input_filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        output_filepath = os.path.join(self.output_dir_path, 'output.png')
        cv2.imwrite(output_filepath, image_with_mask)

    def handle(self, file) -> None:
        temp_name = self._save_temp_file(file)
        image, image_with_mask = self.service.recognize_oil_spill(temp_name)
        self._save_recognition_results(image, image_with_mask)
