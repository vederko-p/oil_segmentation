
import os
from tempfile import NamedTemporaryFile

import numpy as np
import cv2
from loguru import logger

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from infrastructure.segmentation import DummySegmentation
from infrastructure.read_video import read_video
from infrastructure.image_slicing import crop_image_by_edges


app = FastAPI()
templates = Jinja2Templates(directory="templates")


# Init models
# init recognition service
# init handler


segmentation_model = DummySegmentation('path/to/weights')


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    context = {'request': request}
    return templates.TemplateResponse('upload_file.html', context)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):

    logger.info('Saving temporary file')
    temp = NamedTemporaryFile(delete=False)
    contents = await file.read()
    with temp as f:
        f.write(contents)

    logger.info('Processing video')
    video_frames = read_video(temp.name)
    os.remove(temp.name)
    del contents

    logger.info('Stitching frames')
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    status, output = stitcher.stitch(video_frames)
    if status != 0:
        logger.warning('Cant stitch frames into single image')
        # TODO: Handle stitching exception

    logger.info('Oil Spills segmentation process')
    output = crop_image_by_edges(output).copy()
    oil_spill_mask = segmentation_model.segment(output)
    h, w, c = output.shape
    res = (
            np.stack([oil_spill_mask for _ in range(c)]).transpose(1, 2, 0)
            * np.ones((h, w, c)) * np.array([255, 0, 0])
    ).astype(int)
    res += output
    res = np.where(res > 255, 255, res) * (output > 0)

    return {"filename": file.filename}

    # segmentation_model = DummySegmentation('path/to/weights')
    # test_img = np.random.random((100, 100, 1))
    # res = segmentation_model.segment(test_img)
    # print(res.shape)

    # body =
    # result =
    # logging
    # return
