import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse

from .pipeline import handle_image
from .ai.ai import ModelWrapper

app = FastAPI()
ai = ModelWrapper(
    layout_model_name='microsoft/layoutlmv3-large',
    ocr_model_name='microsoft/trocr-large-handwritten',
    ner_model_type='bert',
    ner_model_name='../models/ner/ner',
    timeout=60
)

@app.get('/')
async def root():
    return {'message': 'Hello from ASLA'}


@app.get('/ping')
async def ping():
    return PlainTextResponse(content="OK", status_code=200)

@app.post('/image/')
async def upload_image(name: str = Form(...), image: UploadFile = File(...)):
    print('name:', name)
    print('image.filename:', image.filename)
    # read the file:
    contents = await image.read()
    raw_image = np.frombuffer(contents, np.uint8)
    metadata = handle_image(raw_image, ai)
    return metadata


@app.post('/retrain/')
async def handle_retrain(metadata: str = Form(...), image: UploadFile = File(...)):
    # Safe image and metadata to /data/retraining/
    pass


@app.post('/dev/image/')
async def handle_image_dev(image: UploadFile = File(...)):
    print('received image:', image.filename)
    from asyncio import sleep
    await sleep(3)
    return [
        {
            'label': 'loc',
            'text': 'WETTINGEN',
            'hasBoundingBox': True,
            'boundingBox': {
                'top': 399,
                'right': 3031,
                'bottom': 485,
                'left': 2532
            },
            'manuallyChanged': False
        },
        {
            'label': 'mst',
            'text': 'M. 1:100',
            'hasBoundingBox': True,
            'boundingBox': {
                'top': 407,
                'right': 3744,
                'bottom': 492,
                'left': 3395
            },
            'manuallyChanged': False
        },
        {
            'label': 'date',
            'text': '16.2.40',
            'hasBoundingBox': True,
            'boundingBox': {
                'top': 3545,
                'right': 2560,
                'bottom': 3616,
                'left': 2268
            },
            'manuallyChanged': False
        },
        {
            'label': 'date',
            'text': '1419',
            'hasBoundingBox': True,
            'boundingBox': {
                'top': 3609,
                'right': 2525,
                'bottom': 3680,
                'left': 2361
            },
            'manuallyChanged': False
        },
    ]

