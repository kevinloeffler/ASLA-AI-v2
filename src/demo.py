
""" Run the pipeline without the webserver """

import datetime
import cv2
import numpy as np

from pipeline import handle_image
from preprocessing.image_processing import preprocess_image
from src.ai.ai import ModelWrapper

if __name__ == '__main__':
    # image_path = 'test_image.JPG'
    image_path = '../CRE_test.jpg'
    # image_path = '../data/test_images/KLA_6533_00002.JPG'
    # image_path = cv2.imread('../data/test_images/MN_1595_3.jpg')
    # image_path = cv2.imread('../data/test_images/GM.10163.1.1.JPG')

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ai = ModelWrapper(
        layout_model_name='microsoft/layoutlmv3-large',
        ocr_model_name='microsoft/trocr-large-handwritten',
        ner_model_type='bert',
        ner_model_name='../models/v1',
        timeout=60
    )

    _, buffer = cv2.imencode('.jpg', image)
    image_data = np.frombuffer(buffer, dtype=np.uint8)

    start = datetime.datetime.now()

    metadata = handle_image(image_data, ai)
    print(metadata)

    end = datetime.datetime.now()
    print('handle image timing:', end - start)
