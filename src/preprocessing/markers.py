import cv2
import numpy as np


class Marker:
	def __init__(self, x: int, y: int, width: int, height: int):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.center = (x + int(width / 2), y + int(height / 2))

	def __repr__(self):
		# return f"{self.center}"
		return f"({self.x}, {self.y})"


class Markers:
	"""
	Finds markers in an image and provides helpful methods to work with them.
	:var top_left: top left corner of the marker
	:var top_right: top left corner of the marker
	:var bottom_right: bottom right corner of the marker
	:var bottom_left: bottom left corner of the marker
	"""

	def __init__(self, image: np.ndarray, confidence=0.2):
		markers = self._find_markers(image, confidence)
		self.top_left, self.top_right, self.bottom_right, self.bottom_left = self.assign_markers(markers=markers)

	def __repr__(self):
		return f'Markers: {self.top_left}, {self.top_right}, {self.bottom_right}, {self.bottom_left}'

	def all(self) -> list[Marker]:
		return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

	def _find_markers(self, image: np.ndarray, threshold: float) -> list[Marker]:
		template = cv2.imread('preprocessing/asla-marker.jpg')
		template_height, template_width = template.shape[:-1]

		match_results = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
		matches = []
		selection = []

		for y, row in enumerate(match_results):
			for x, score in enumerate(row):
				if score > threshold:
					matches.append((x, y, score))

		for i in range(4):
			max_index = np.argmax([match[2] for match in matches])
			selection.append(matches[max_index][:2])
			max_element = matches.pop(max_index)
			new_matches = filter(lambda match: not self._is_close((max_element[0], max_element[1]), match[:2], template_width), matches)
			matches = list(new_matches)

		print('selection:', selection)

		print('close:', self._is_close((663, 384), (662, 384), template_width))

		# visualize result:
		# for match in selection:
		# 	cv2.rectangle(image, (match[0], match[1]), (match[0] + template_width, match[1] + template_height), (0, 0, 255), 2)
		# cv2.imshow('preview', image)
		# cv2.waitKey(0)

		if len(selection) < 4:
			raise MarkerError(f'could not find 4 markers above the confidence threshold: {threshold}')

		return [Marker(x=match[0], y=match[1], width=template_width, height=template_height) for match in selection]

	@staticmethod
	def _is_close(a: tuple[int, int], b: tuple[int, int], threshold: int) -> bool:
		delta_x, delta_y = b[0] - a[0], b[1] - a[1]
		return abs(delta_x) < threshold and abs(delta_y) < threshold

	@staticmethod
	def assign_markers(markers: list[Marker]) -> tuple[Marker, Marker, Marker, Marker]:
		# Sort markers based on x-coordinate
		sorted_markers = sorted(markers, key=lambda marker: marker.x)

		# Sort left markers based on y-coordinate
		left_markers = sorted(sorted_markers[:2], key=lambda marker: marker.y)
		right_markers = sorted(sorted_markers[2:], key=lambda marker: marker.y)

		top_left: Marker = left_markers[0]
		bottom_left: Marker = left_markers[1]
		top_right: Marker = right_markers[0]
		bottom_right: Marker = right_markers[1]

		return top_left, top_right, bottom_right, bottom_left


class MarkerError(Exception):
	def __init__(self, msg):
		super().__init__(msg)
