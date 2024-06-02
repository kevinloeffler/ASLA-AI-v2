import torch
import numpy as np
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs


class NerModel:
	def __init__(
			self, model_type: str, model_name: str,
			training_iterations: int = 1, safe_to: str = None,
			numbers_of_gpus: int = 0, gpu_id: int = 0
	):

		self.__has_cuda = torch.cuda.is_available()
		use_cuda = self.__has_cuda

		self.labels = ['O', 'CLT', 'LOC', 'MST', 'CLOC', 'DATE']

		model_args = NERArgs()
		model_args.labels_list = self.labels
		model_args.num_train_epochs = training_iterations
		model_args.classification_report = True
		model_args.use_multiprocessing = True
		model_args.save_model_every_epoch = False
		# model_args.output_dir = safe_to or ''
		if numbers_of_gpus > 0:
			use_cuda = True
			model_args.n_gpu = numbers_of_gpus

		self.model = NERModel(model_type, model_name, use_cuda=use_cuda, cuda_device=gpu_id, args=model_args)

	def predict(self, text: str) -> list[dict]:
		print('NER predicting:', text)
		prediction, outputs = self.model.predict([text])
		return prediction[0]


if __name__ == '__main__':
	model = NerModel('bert', '../../../models/v1', 1)
	prediction = model.predict('Garten des Herrn Gretsch, Wettswil')
	print(prediction)
