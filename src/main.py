import numpy as np
from fastapi import FastAPI, UploadFile, File, Form

from pipeline import handle_image
from .ai.ai import ModelWrapper

app = FastAPI()
ai = ModelWrapper(
    layout_model_name='microsoft/layoutlmv3-large',
    ocr_model_name='microsoft/trocr-large-handwritten',
    ner_model_type='bert',
    ner_model_name='../models/v1',
    timeout=60
)

@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/image/')
async def upload_image(name: str = Form(...), image: UploadFile = File(...)):
    print('name:', name)
    print('image.filename:', image.filename)
    # read the file:
    contents = await image.read()
    raw_image = np.frombuffer(contents, np.uint8)
    metadata = handle_image(raw_image, ai)
    return metadata
