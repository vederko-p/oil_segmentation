
import os
from tempfile import NamedTemporaryFile

import numpy as np
from loguru import logger

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from infrastructure.segmentation import DummySegmentation
from infrastructure.read_video import read_video


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
    print(f'video_frames: {video_frames.shape}')

    return {"filename": file.filename}

    # segmentation_model = DummySegmentation('path/to/weights')
    # test_img = np.random.random((100, 100, 1))
    # res = segmentation_model.segment(test_img)
    # print(res.shape)

    # body =
    # result =
    # logging
    # return
