
import os

from loguru import logger

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.infrastructure.segmentation import DummySegmentation

from src.handlers.recognition import RecognitionHandler
from src.service.recognition import SegmentationService


# Config:
OUTPUT_DIR_PATH = './src/static'
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)


app = FastAPI()
app.mount("/src/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")


# Init model
segmentation_model = DummySegmentation('path/to/weights')

# Init service
segmentation_service = SegmentationService(segmentation_model)

# Init handler
recognition_handler = RecognitionHandler(
    segmentation_service,
    OUTPUT_DIR_PATH
)


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    context = {'request': request}
    return templates.TemplateResponse('upload_file.html', context)


@app.post("/uploadfile/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    recognition_handler.handle(file)
    logger.info('Showing segmentation results')
    j_input_filepath = os.path.join('/src/static', 'input.png')
    j_output_filepath = os.path.join('/src/static', 'output.png')
    context = {
        'input_filepath': j_input_filepath,
        'output_filepath': j_output_filepath,
        'request': request,
    }
    return templates.TemplateResponse('segm_results.html', context)
