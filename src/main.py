
import numpy as np

from infrastructure.segmentation import DummySegmentation


if __name__ == '__main__':
    segmentation_model = DummySegmentation('path/to/weights')
    test_img = np.random.random((100, 100, 1))
    res = segmentation_model.segment(test_img)
    print(res.shape)

