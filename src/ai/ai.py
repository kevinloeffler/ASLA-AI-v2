import datetime
import os

import cv2
from transformers import LayoutLMv3Processor

from src.ai.layout.layout_model import LayoutModel
from src.ai.layout.layout_model_test import layout_benchmark
from src.preprocessing.metadata import read_metadata


class ModelWrapper:

	layout_model = None
	ocr_model = None
	ner_model = None

	def __init__(self, layout_model: str, ocr_model: str, ner_model: str):
		self.layout_model = LayoutLMv3Processor.from_pretrained(layout_model)
		# self.ocr_model = TrOCR(ocr_model)
		# self.knn_model = Clustering()
		# self.ner_model = None
		print('Done loading models')


def extract_entities(image) -> list:

	return [
		{
			'label': 'loc',
			'text': 'WETTINGEN',
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
			'boundingBox': {
				'top': 3609,
				'right': 2525,
				'bottom': 3680,
				'left': 2361
			},
			'manuallyChanged': False
		},
	]


def test_models(layout_model_name: str, ocr_model_name: str, ner_model_name: str,
				directory: str, output_dir: str, comments: any = ''):
	layout_model = LayoutModel('microsoft/layoutlmv3-large')
	# ocr_model =
	# ner_model =

	layout_results = []
	ocr_results = []
	ner_results = []

	test_image_count = 0

	for file in os.listdir(directory):
		if file.endswith('.json'):
			try:
				image_filename = file.replace('.json', '.jpg')
				image = cv2.imread(os.path.join(directory, image_filename))
				metadata = read_metadata(os.path.join(directory, file))

				test_image_count += 1

				# do preprocessing...

				layout_results.append(layout_benchmark(layout_model, image, metadata['entities']))

			except Exception as e:
				print(e)  # TODO: error handling


	layout_performance = sum(layout_results) / len(layout_results)

	output_string = f"""
	BENCHMARK REPORT
	
	---------------- Config
	layout model: {layout_model_name}
	ocr model:    {ocr_model_name}
	ner model:    {ner_model_name}
	test images:  {test_image_count}
	
	---------------- Results
	layout model: {round(layout_performance, 3)}
	ocr model:    ?
	ner model:    ?
	pipeline:     ?
	
	---------------- Comments
	{comments}
	"""

	now = datetime.datetime.now()
	filename = f"benchmark_report_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}.txt"
	with open(os.path.join(output_dir, filename), 'w') as file:
		file.write(output_string)

	return {
		"layout_model": layout_performance,
		"ocr_model": 0,
		"ner_model": 0,
		"pipeline": 0,
	}
