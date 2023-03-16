
import cv2
import numpy as np


def check_cap(capture):
    if not capture.isOpened():
        raise Exception('Error opening video stream or file.')


def init_frames_storage(cap):
    _, frame = cap.read()
    return np.expand_dims(frame, 0)


def read_video(video_path, step=10):
    """
    Returns batch of BGR frames with shape (n, h, w, c).
    """
    cap = cv2.VideoCapture(video_path)
    check_cap(cap)
    frames_storage = init_frames_storage(cap)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_id += 1
        if not ret:
            break
        if not (frame_id % step):
            frames_storage = np.append(
                frames_storage, np.expand_dims(frame, 0), axis=0
            )
    cap.release()
    return frames_storage
