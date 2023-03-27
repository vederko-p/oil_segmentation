import os
import time
import numpy as np

import cv2
import streamlit as st

from ultralytics import YOLO

from read_video import read_video


INPUT_PATH = "../files/input/"
OUTPUT_PATH = "../files/output/"

model = YOLO("../notebooks/yolo/weights/s_v3.pt")

st.title("Oil spill segmentation module")

input_file = st.file_uploader("Upload video", type=[".mp4"])

if input_file is not None:
    st.write("Video is processing, please wait...")
    with open(INPUT_PATH + input_file.name, mode='wb') as w:
        w.write(input_file.getvalue())

    frames = read_video(INPUT_PATH + input_file.name, step=24)

    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, output = stitcher.stitch(frames)
    if status == 0:
        st.write("Frames successfully stitched into one image")
    else:
        st.write(f'Error while stiching, status: {status}')
    cv2.imwrite(OUTPUT_PATH + input_file.name[:-4] + ".jpg", output)

    orig = output.copy()
    h, w, _ = output.shape

    predictions = model.predict(output)
    cv2.imwrite(OUTPUT_PATH + input_file.name[:-4] + "_seg.jpg", output)

    size = cv2.countNonZero(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY))

    mask = np.zeros((h, w))
    xy = [(object.masks.segments[0] * np.array([[w, h]])).astype(int) for object in predictions]
    cv2.fillPoly(mask, xy, (255, 255, 255))
    cv2.imwrite(OUTPUT_PATH + input_file.name[:-4] + "_mask.jpg", mask)
    mask_size = cv2.countNonZero(mask)

    spill_size = int(mask_size/size * 100)
    st.write(f'Oil spill covers {spill_size} percent of the water surface')

    vis = np.concatenate((orig, output), axis=1)
    st.image(vis, channels='RGB', caption="TEST")
