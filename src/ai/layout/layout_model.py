import cv2
import numpy
from transformers import LayoutLMv3ImageProcessor
from dataclasses import dataclass

from .clustering import Clustering
from ...util.types import BoundingBox


class LayoutModel:

	def __init__(self, model: str):
		self.processor = LayoutLMv3ImageProcessor.from_pretrained(model)  # could add: ocr_lang='deu')
		self.knn_model = Clustering()

	def predict(self, raw_image: numpy.ndarray, overlap_threshold=0.8) -> list[list[BoundingBox]]:
		"""Takes an open cv image (BGR), predicts all the bounding boxes and returns groups of all detected words"""

		# Predict
		image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
		encoding = self.processor(image, return_tensors="pt")
		del encoding['pixel_values']

		# Get bounding boxes
		image_size = (image.shape[1], image.shape[0])
		bounding_boxes = self.__get_bounding_boxes(encoding, image_size=image_size, overlap_threshold=overlap_threshold)
		ordered_boxes = self.__order_boxes(bounding_boxes)

		# Draw boxes (for visual debugging)
		# self.__draw_boxes(image, ordered_boxes)

		return ordered_boxes


	def __get_bounding_boxes(self, encoding, image_size: tuple[int, int], overlap_threshold: float) -> list[list[BoundingBox]]:
		"""Normalize, combine and group bounding boxes"""
		image_width, image_height = image_size
		normalized_boxes = list(set([(box[0], box[1], box[2], box[3]) for box in encoding['boxes'][0]]))
		all_boxes = [self.__unnormalize_box(box, image_width, image_height) for box in normalized_boxes]
		overlapping_boxes = list(filter(lambda box: not self.__is_small_box(box, min_axis_length=16), all_boxes))
		ungrouped_boxes = self.__combine_overlapping_boxes(overlapping_boxes, overlap_threshold)
		if len(ungrouped_boxes) == 0:
			raise ValueError('ERROR: Layout model did not find any boxes')
		return self.knn_model.predict_groups(ungrouped_boxes)

	def __order_boxes(self, boxes_groups: list[list[BoundingBox]]) -> list[list[BoundingBox]]:
		"""Bring all bounding boxes of a group into normal reading order (top to bottom, left to right)"""
		ordered_boxes = []
		for boxes in boxes_groups:
			rows: list[BoxRow] = []
			for box in boxes:
				self.__find_boxes_in_same_row(box, rows)

			sorted_rows = self.__sort_boxes(rows)
			ordered_boxes.append(sorted_rows)
		return ordered_boxes

	@staticmethod
	def __find_boxes_in_same_row(box, rows):
		"""Combine boxes with a similar y value into a BoxRow"""
		if len(rows) == 0:
			rows.append(BoxRow(upper_bound=box[1], lower_bound=box[3], children=[box]))
			return

		for row in rows:
			box_center = box[1] + round((box[3] - box[1]) / 2)
			if row.upper_bound < box_center < row.lower_bound:
				row.add(box)
				return
		rows.append(BoxRow(upper_bound=box[1], lower_bound=box[3], children=[box]))
		return

	@staticmethod
	def __sort_boxes(rows: list[any]) -> list[BoundingBox]:
		"""Sort all boxes in a row / line from left to right according to the left x value"""
		sorted_boxes = []
		# sort lines (rows) vertically
		vertically_sorted_rows = list(sorted(rows, key=lambda row: row.upper_bound))
		# sort words in line left to right
		for row in vertically_sorted_rows:
			sorted_row = list(sorted(row.children, key=lambda box: box[0]))
			sorted_boxes += sorted_row
		return sorted_boxes

	@staticmethod
	def __unnormalize_box(bbox, width: int, height: int) -> BoundingBox:
		"""The LayoutLM model returns normalized (0-100) bounding boxes which need to be scaled back to normal"""
		return (
			round(width * (bbox[0] / 1000)),
			round(height * (bbox[1] / 1000)),
			round(width * (bbox[2] / 1000)),
			round(height * (bbox[3] / 1000)),
		)

	@staticmethod
	def __resize_bounding_box(bounding_box: BoundingBox, by: int, image_size: tuple[int, int]) -> BoundingBox:
		"""Add a 'safety' margin to bounding boxes to handle slight prediction errors"""
		return (
			max(bounding_box[0] - by, 0),
			max(bounding_box[1] - by, 0),
			min(bounding_box[2] + by, image_size[0]),
			min(bounding_box[3] + by, image_size[1]),
		)

	def __combine_overlapping_boxes(self, boxes: list[BoundingBox], overlap_threshold: float) -> list[BoundingBox]:
		"""Combine boxes that overlap a certain amount into one"""
		combined_boxes = []

		for i, box1 in enumerate(boxes):
			skip_box = False
			for j, box2 in enumerate(combined_boxes):
				if self.__iom(box1, box2) > overlap_threshold:
					# Merge the boxes if the overlap is above the threshold
					combined_boxes[j] = (
						min(box1[0], box2[0]),
						min(box1[1], box2[1]),
						max(box1[2], box2[2]),
						max(box1[3], box2[3]),
					)
					skip_box = True
					break

			if not skip_box:
				combined_boxes.append(box1)

		return combined_boxes

	@staticmethod
	def __iom(box1: BoundingBox, box2: BoundingBox) -> float:
		"""Calculate the Intersection over Minimum (IoM) of two bounding boxes"""
		x1 = max(box1[0], box2[0])
		y1 = max(box1[1], box2[1])
		x2 = min(box1[2], box2[2])
		y2 = min(box1[3], box2[3])

		intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
		min_area = min((box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1]))

		iom = intersection_area / min_area if min_area > 0 else 0
		return iom

	@staticmethod
	def __is_small_box(box: BoundingBox, min_axis_length: int = 10):
		x_axis = box[2] - box[0]
		y_axis = box[3] - box[1]
		return x_axis < min_axis_length or y_axis < min_axis_length

	@staticmethod
	def __draw_boxes(image, grouped_boxes: list[list[BoundingBox]]):
		"""draw predictions over the image"""
		colors = [(244, 159, 10), (239, 62, 54), (20, 138, 206), (12, 124, 89), (214, 73, 51), (11, 157, 82)]

		# print('boxes:', grouped_boxes)
		for index, group in enumerate(grouped_boxes):
			group_color = colors[index % len(colors)]
			for box in group:
				cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), group_color[::-1], 3)
		cv2.imshow('preview', image)
		cv2.waitKey(0)


# Helper classes

@dataclass
class BoxRow:
	"""Used to find bounding boxes in the same row"""
	upper_bound: int
	lower_bound: int
	children: list[BoundingBox]

	def add(self, box: BoundingBox):
		self.children.append(box)
		self.upper_bound = min(box[1], self.upper_bound)
		self.lower_bound = max(box[3], self.lower_bound)

	def get_center(self) -> int:
		return round((self.lower_bound - self.upper_bound) / 2) + self.upper_bound

