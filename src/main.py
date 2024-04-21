# import torch
import math

# import numpy as np
# import pandas as pd
# from simpletransformers.ner import NERModel, NERArgs
import cv2

from preprocessing.format import format_image
from preprocessing.markers import Markers
from preprocessing.metadata import create_json_metadata


# image
test_image = cv2.imread('test_image.JPG')
image_height, image_width = test_image.shape[:2]

# format
markers = Markers(test_image)
crop, angle = format_image(markers=markers, image_shape=(image_width, image_height))

# create metadata
metadata = create_json_metadata(
	entities=[],
	contrast=1.0,
	brightness=1.0,
	sharpness=1.0,
	white_balance=[1.0, 1.0, 1.0],
	crop_top=crop[0],
	crop_right=crop[1],
	crop_bottom=crop[2],
	crop_left=crop[3],
	rotation=math.degrees(angle),
)

print(metadata)

'''
print('pytorch version:', torch.__version__)

model_args = NERArgs()
model_args.labels_list = ['O', 'CLT', 'LOC', 'MST', 'CLOC', 'DATE']
model_args.num_train_epochs = 1
model_args.classification_report = True
model_args.use_multiprocessing = True
model_args.save_model_every_epoch = False
# model_args.output_dir = safe_to

model = NERModel('bert', 'models/v1', use_cuda=False, args=model_args)

prediction = model.predict(['Garten des Herrn Gretsch, Wettswil'])[0]

print(prediction)
'''