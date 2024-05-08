import numpy as np
from fastapi import FastAPI, UploadFile, File, Form

from pipeline import handle_image, demo

app = FastAPI()


@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.post('/image/')
async def upload_image(name: str = Form(...), image: UploadFile = File(...)):
    print('name:', name)
    print('image.filename:', image.filename)
    # read the file:
    contents = await image.read()
    np_array = np.frombuffer(contents, np.uint8)
    metadata = handle_image(np_array)
    return metadata
