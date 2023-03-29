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

input_file = st.file_uploader("Upload image or video", type=[".mp4", ".jpg", ".png"])

if input_file is not None:
    with open(INPUT_PATH + input_file.name, mode='wb') as w:
        w.write(input_file.getvalue())

    if ".mp4" in input_file.name:
        info = st.info("Processing video, please wait")
        progress_bar = st.progress(0, text="Stitching...")

        frames = read_video(INPUT_PATH + input_file.name, step=24)

        stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        status, output = stitcher.stitch(frames)
        if status == 0:
            progress_bar.progress(0.33, text="Processing...")
        else:
            st.error(f'Error while stitching, status: {status}')
        cv2.imwrite(OUTPUT_PATH + input_file.name[:-4] + ".jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    if any(ext in input_file.name for ext in [".jpg", ".png"]):
        info = st.info("Processing picture, please wait")
        progress_bar = st.progress(0.33, text="Processing...")
        output = cv2.imread(INPUT_PATH + input_file.name)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.imwrite(OUTPUT_PATH + input_file.name[:-4] + ".jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    orig = output.copy()
    h, w, _ = output.shape

    predictions = model.predict(output)
        
    cv2.imwrite(OUTPUT_PATH + input_file.name[:-4] + "_seg.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    size = cv2.countNonZero(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY))

    progress_bar.progress(0.65, text="Calculating the area...")

    for object in predictions:
        if object.masks is not None:
            mask = np.zeros((h, w))
            xy = [(object.masks.segments[0] * np.array([[w, h]])).astype(int)]
            cv2.fillPoly(mask, xy, (255, 255, 255))
            cv2.imwrite(OUTPUT_PATH + input_file.name[:-4] + "_mask.jpg", mask)
            mask_size = cv2.countNonZero(mask)

            spill_size = int(mask_size/size * 100)

            info.empty()
            st.success("Done!")
            progress_bar.progress(0.99, text="Calculating the area...")
            st.write(f'Oil spill covers {spill_size} percent of the water surface')
            progress_bar.empty()

            vis = np.concatenate((orig, output), axis=1)
            st.image(vis, channels='RGB', caption="TEST")
        else:
            st.success("Done!")
            st.write(f'There is no oil spill')
            info.empty()
            progress_bar.empty()
            st.image(orig, channels='RGB', caption="TEST")

