
import os

import cv2
import numpy as np


TEST_VIDEO_PATH = os.path.join(
    '/home/maksim/Desktop/ITMO Lectures/3. CV DL and CV',
    'Code/test _video/video 3/video_3_1.mp4'
)


def check_cap(capture):
    if not capture.isOpened():
        raise Exception('Error opening video stream or file')


def read_video(video_path, show=False):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_video(TEST_VIDEO_PATH, show=True)
