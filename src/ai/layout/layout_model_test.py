"""
Benchmark / test the performance of the layout model.

Two metrics:
1. Entities located, measured in % of boxes
2. All boxes, F1 score

Expects the target entities to be in the following format:
{
	"label": "loc",
	"text": "WETTINGEN",
	"boundingBox": {
		"top": 399,
		"right": 3031,
		"bottom": 485,
		"left": 2532
	}
	...
}
The origin of the coordinate system is in the top left
"""
import os

import cv2
from numpy import ndarray

from .layout_model import LayoutModel
from ...preprocessing.metadata import read_metadata


def layout_benchmark(layout_model: LayoutModel, image: ndarray, target_entities: list[dict[str, any]]) -> float:
	prediction_result = layout_model.predict(image)
	predicted_boxes = [box for boxes in prediction_result for box in boxes]

	entities = 0
	predicted_entities = 0

	for target_entity in target_entities:
		entities += 1
		for predicted_box in predicted_boxes:
			if __compare_boxes(target_entity, predicted_box):
				# print('match:', predicted_box, target_entity)
				predicted_entities += 1
				break

	print(f'predicted {predicted_entities} / {entities} entities correctly')
	return predicted_entities / entities


def __compare_boxes(target, prediction, threshold=20) -> bool:
	"""Compare if the predicted box is similar to a target entity"""
	'''Problems: Predicted boxes that are very close to each other are not counted as correct (could be intentional)'''
	return (
			abs(int(target['boundingBox']['left']) - prediction[0]) <= threshold and
			abs(int(target['boundingBox']['top']) - prediction[1]) <= threshold and
			abs(int(target['boundingBox']['right']) - prediction[2]) <= threshold and
			abs(int(target['boundingBox']['bottom']) - prediction[3]) <= threshold
	)


def run_layout_benchmark(model: LayoutModel, directory='/data/test_images/'):
	results = []
	for file in os.listdir(directory):
		if file.endswith('.json'):
			try:
				image_filename = file.replace('.json', '.jpg')
				image = cv2.imread(os.path.join(directory, image_filename))
				metadata = read_metadata(os.path.join(directory, file))
				results.append(layout_benchmark(model, image, metadata['entities']))
			except ValueError as e:
				print('Model did not find any boxes in file: {file}')
				results.append(0)

	performance = sum(results) / len(results)
	print(f'-------------------------------')
	print(f'Performance: {performance}')
	print(f'-------------------------------')
	return performance

