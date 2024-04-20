# import torch
import numpy as np
# import pandas as pd
# from simpletransformers.ner import NERModel, NERArgs
import cv2

from src.preprocessing.format import format_image, rotate_marker
from src.preprocessing.markers import Markers, Marker


test_image = cv2.imread('test_image.JPG')
image_height, image_width = test_image.shape[:2]

# rotation test:
# res = rotate_marker(Marker(x=100, y=100, width=100, height=100), 0.178, (image_width, image_height))
# print('res:', res)

image_center = tuple(np.array(test_image.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D(image_center, -1.64, 1.0)
rotated_image = cv2.warpAffine(test_image, rot_mat, test_image.shape[1::-1], flags=cv2.INTER_LINEAR)

markers = Markers(test_image)
print('markers:', markers)
t, r, b, l = format_image(markers=markers, image_shape=(image_width, image_height))

for marker in markers.all():
	rotated_marker = rotate_marker(marker, -0.178, (image_width, image_height))
	cv2.rectangle(rotated_image, rotated_marker, (rotated_marker[0] + 10, rotated_marker[1] + 10), (0, 255, 0), 2)

# visualize result:
cv2.rectangle(rotated_image, (l, t), (image_width - r, image_height - b), (0, 0, 255), 2)
cv2.imshow('preview', rotated_image)
cv2.waitKey(0)

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