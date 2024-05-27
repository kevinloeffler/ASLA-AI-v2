from difflib import SequenceMatcher

import cv2
from numpy import ndarray

from src.ai.ocr.ocr_model import OcrModel
from src.preprocessing.metadata import read_metadata


def ocr_benchmark(ocr_model: OcrModel, image: ndarray, target: str) -> float:
	prediction = ocr_model.predict(image) or ''
	target = OcrModel.post_process_prediction(target)
	similarity = SequenceMatcher(None, prediction, target).ratio() ** 2
	print(f'ocr score: {round(similarity, 3)} | "{prediction}" vs "{target}"')
	return similarity


def run_ocr_benchmark(ocr_model: OcrModel, image_path: str, metadata_path: str):
	image = cv2.imread(image_path)
	metadata = read_metadata(metadata_path)
	for entity in metadata['entities']:
		cropped_image = image[
						entity['boundingBox']['top']: entity['boundingBox']['bottom'],
						entity['boundingBox']['left']: entity['boundingBox']['right']]
		ocr_benchmark(ocr_model, cropped_image, entity['text'])
