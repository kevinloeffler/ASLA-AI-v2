import cv2


class Markers:
	"""
	Finds markers in an image and provides helpful methods to work with them.
	:var top_left: top left corner of the marker
	:var top_right: top left corner of the marker
	:var bottom_right: bottom right corner of the marker
	:var bottom_left: bottom left corner of the marker
	"""

	def __init__(self, for_image: str, confidence=0.2):
		markers = self._find_markers(for_image, confidence)
		self.top_left, self.top_right, self.bottom_right, self.bottom_left = self.assign_markers(markers=markers)

	def __repr__(self):
		return f'Markers: {self.top_left}, {self.top_right}, {self.bottom_right}, {self.bottom_left}'

	def _find_markers(self, image_path: str, threshold: float):
		image = cv2.imread(image_path)
		template = cv2.imread('preprocessing/asla-marker.jpg')
		template_width, template_height = template.shape[:-1]

		match_results = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

		best_matches = [(-1, -1000, -1000), (-1, -1000, -1000), (-1, -1000, -1000), (-1, -1000, -1000)]  # (score, x, y)

		for y, row in enumerate(match_results):
			for x, score in enumerate(row):
				if score > best_matches[0][0]:
					done = False
					for index, match in enumerate(best_matches):
						if self._is_close((x, y), (match[1], match[2]), template_width):
							if score > match[0]:
								best_matches[index] = (score, x, y)
								done = True
								break
							else:
								done = True
								break

					if not done:
						best_matches[0] = (score, x, y)

					best_matches.sort(key=lambda t: t[0])
		# visualize result:
		# for match in best_matches:
		# 	cv2.rectangle(image, (match[1], match[2]), (match[1] + template_width, match[2] + template_height), (0, 0, 255), 2)
		# cv2.imshow('preview', image)
		# cv2.waitKey(0)
		if any([marker[0] < threshold for marker in best_matches]):
			raise MarkerError(f'could not find 4 markers in {image_path} above confidence threshold: {threshold}')

		return best_matches

	@staticmethod
	def _is_close(a: tuple[int, int], b: tuple[int, int], threshold: int) -> bool:
		delta_x, delta_y = b[0] - a[0], b[1] - a[1]
		return abs(delta_x) < threshold and abs(delta_y) < threshold

	@staticmethod
	def assign_markers(markers: list[tuple[int, int, int]]) -> tuple:
		# Sort markers based on x-coordinate
		sorted_markers = sorted(markers, key=lambda marker: marker[1])

		# Determine left and right markers
		left_markers = sorted_markers[:2]
		right_markers = sorted_markers[2:]

		# Sort left markers based on y-coordinate
		top_left = min(left_markers, key=lambda marker: marker[2])[1:]
		bottom_left = max(left_markers, key=lambda marker: marker[2])[1:]

		# Sort right markers based on y-coordinate
		top_right = min(right_markers, key=lambda marker: marker[2])[1:]
		bottom_right = max(right_markers, key=lambda marker: marker[2])[1:]

		return top_left, top_right, bottom_right, bottom_left


class MarkerError(Exception):
	def __init__(self, msg):
		super().__init__(msg)
