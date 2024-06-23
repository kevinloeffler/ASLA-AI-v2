import datetime
import json
import os
import threading

import cv2
import numpy as np

from src.ai.layout.layout_model import LayoutModel
from src.ai.layout.layout_model_test import layout_benchmark
from src.ai.ner.ner_model import NerModel
from src.ai.ner.ner_model_test import ner_benchmark
from src.ai.ocr.ocr_model import OcrModel
from src.ai.ocr.ocr_model_test import ocr_benchmark
from src.postprocessing.ai_postprocessing import postprocess
from src.preprocessing.image_processing import preprocess_image
from src.preprocessing.metadata import read_metadata
from src.util.types import PredictedEntity
from src.util.util import find_best_label


class ModelWrapper:
	def __init__(self, layout_model_name: str, ocr_model_name: str, ner_model_name: str, ner_model_type: str, timeout=60*30):
		self.model_lock = threading.Lock()
		self.timer = None
		self.timeout = timeout

		self.layout_model = None
		self.ocr_model = None
		self.ner_model = None

		self.layout_model_name = layout_model_name
		self.ocr_model_name = ocr_model_name
		self.ner_model_name = ner_model_name
		self.ner_model_type = ner_model_type

		self.load_models()

	def load_models(self):
		print('Loading models')
		self.layout_model = LayoutModel(self.layout_model_name)
		self.ocr_model = OcrModel(self.ocr_model_name)
		self.ner_model = NerModel(self.ner_model_type, self.ner_model_name)
		print('Done loading models')
		self.reset_timer()

	def unload_models(self):
		print('Unloading models')
		self.layout_model = None
		self.ocr_model = None
		self.ner_model = None

	def reset_timer(self):
		print('Resetting timer')
		if self.timer:
			self.timer.cancel()
		self.timer = threading.Timer(self.timeout, self.unload_models)
		self.timer.start()

	def run_pipeline(self, image: np.ndarray) -> list[list[PredictedEntity]]:
		with self.model_lock:
			if self.layout_model is None:
				self.load_models()

			self.timer.cancel()
			extracted_entities = extract_entities(image, self.layout_model, self.ocr_model, self.ner_model)
			self.reset_timer()
			return extracted_entities


def extract_entities(image: np.ndarray, layout_model: LayoutModel, ocr_model: OcrModel, ner_model: NerModel) -> list[list[PredictedEntity]]:
	# Layout
	clusters = layout_model.predict(image)

	# OCR
	ocr_clusters = []
	for boxes in clusters:
		cluster_words = []
		for box in boxes:
			cropped_image = image[box[1]-5: box[3]+5, box[0]-5: box[2]+5]
			text, confidence = ocr_model.predict(cropped_image)
			predicted_word: PredictedEntity = {
				'label': None,
				'text': text,
				'boundingBox': box,
				'ocr_confidence': confidence,
			}
			cluster_words.append(predicted_word) if text else None
		ocr_clusters.append(cluster_words)

	# NER
	for cluster in ocr_clusters:
		sentence = ' '.join(word['text'] for word in cluster)
		local_entities = ner_model.predict(sentence)
		print('local entities:', local_entities)

		for prediction in cluster:
			prediction['label'] = find_best_label(prediction['text'], local_entities)

	return ocr_clusters


def test_models(layout_model_name: str, ocr_model_name: str, ner_model_name: str, ner_model_type: str,
				directory: str, output_dir: str, comments: any = '', safe=True):
	layout_model = LayoutModel(layout_model_name)
	ocr_model = OcrModel(ocr_model_name)
	ner_model = NerModel(ner_model_type, ner_model_name)

	layout_results = []
	ocr_results = []
	ner_results = []

	test_image_count = 0

	for file in os.listdir(directory):
		if file.endswith('.json'):
			try:
				metadata = read_metadata(os.path.join(directory, file))
				image = cv2.imread(os.path.join(directory, file.replace('.json', '.jpg')))
				text = read_metadata(os.path.join(directory, file.replace('.json', '.text')), False)

				test_image_count += 1

				# preprocessing
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				image = preprocess_image(image)

				# layout model:
				layout_results.append(layout_benchmark(layout_model, image, metadata['entities']))

				# ocr model:
				temp_ocr_scores = []
				for entity in metadata['entities']:
					cropped_image = image[
									entity['boundingBox']['top']: entity['boundingBox']['bottom'],
									entity['boundingBox']['left']: entity['boundingBox']['right']]
					temp_ocr_scores.append(ocr_benchmark(ocr_model, cropped_image, entity['text']))
				ocr_results.append(sum(temp_ocr_scores) / len(temp_ocr_scores))

				# ner model:
				for sentence in text['sentences']:
					ner_results.append(ner_benchmark(ner_model, sentence, metadata['entities']))

			except Exception as e:
				print(e)

	layout_performance = sum(layout_results) / len(layout_results)
	ocr_performance = sum(ocr_results) / len(ocr_results)
	ner_performance = sum(ner_results) / len(ner_results)

	if safe:
		output_string = f"""
		BENCHMARK REPORT
		
		---------------- Config
		layout model: {layout_model_name}
		ocr model:    {ocr_model_name}
		ner model:    {ner_model_name}
		test images:  {test_image_count}
		
		---------------- Results
		layout model: {round(layout_performance, 3)}
		ocr model:    {round(ocr_performance, 3)}
		ner model:    {round(ner_performance, 3)}
		
		---------------- Comments
		{comments}
		"""

		output_json = {
			'config': {
				'layout_model': layout_model_name,
				'ocr_model': ocr_model_name,
				'ner_model': ner_model_name,
				'test_image_count': test_image_count,
			},
			'results': {
				'layout_model': layout_performance,
				'ocr_model': ocr_performance,
				'ner_model': ner_performance,
			},
			'comments': comments
		}

		now = datetime.datetime.now()
		filename = f"benchmark_report_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"

		with open(os.path.join(output_dir, f'{filename}.txt'), 'w') as file:
			file.write(output_string)

		with open(os.path.join(output_dir, f'{filename}.json'), 'w') as file:
			file.write(json.dumps(output_json))
	# END: safe

	return {
		"layout_model": layout_performance,
		"ocr_model": ocr_performance,
		"ner_model": ner_performance,
		"pipeline": 0,
	}
