
import cv2
import numpy as np

from loguru import logger


def check_cap(capture):
    if not capture.isOpened():
        raise Exception('Error opening video stream or file.')


def init_frames_storage(cap):
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.expand_dims(frame, 0)


def read_video(video_path, step=24):
    """
    Returns batch of RGB frames with shape (n, h, w, c).
    """
    logger.info('Reading video into frames')
    cap = cv2.VideoCapture(video_path)
    check_cap(cap)
    f_storage = init_frames_storage(cap)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_id += 1
        if not ret:
            break
        if not (frame_id % step):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            f_storage = np.append(f_storage, np.expand_dims(frame, 0), axis=0)
    cap.release()
    return f_storage
