# import torch
# import numpy as np
# import pandas as pd
# from simpletransformers.ner import NERModel, NERArgs
from src.preprocessing.markers import Markers

markers = Markers('test_image.JPG')
print(markers)


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