import math
import numpy as np

from .markers import Markers, Marker


def format_image(markers: Markers, image_shape: tuple[int, int], margin=25) -> tuple[tuple[int, int, int, int], float]:
	"""
	:param markers: The markers corresponding to the image
	:param image_shape: The shape of the image
	:param margin: Pixels to increase the crop of the image
	:return: (crop value tuple (top, right, bottom, left) and clockwise rotation angle in radians)
	"""
	angle = _get_rotation(markers=markers)
	rotated_markers = [rotate_marker(marker, -angle, image_shape) for marker in markers.all()]

	crop_top = min(rotated_markers, key=lambda marker: marker[1])[1] + margin
	crop_bottom = image_shape[1] - max(rotated_markers, key=lambda marker: marker[1])[1] + margin
	crop_left = min(rotated_markers, key=lambda marker: marker[0])[0] + margin
	crop_right = image_shape[0] - max(rotated_markers, key=lambda marker: marker[0])[0] + margin

	return (crop_top, crop_right, crop_bottom, crop_left), angle


def _get_rotation(markers: Markers) -> float:
	"""
	This method returns the angle by which an image should be rotated to get the markers as straight as possible. It
	compares two markers and calculates how tilted the line between them is compared to the corresponding x or y-axis.
	The rotation is the average of the four angles.
	:param markers: The markers corresponding to the image
	:return: The rotation angle in radians
	"""

	alpha = np.arcsin((markers.top_right.y - markers.top_left.y) / (markers.top_right.x - markers.top_left.x))
	beta = np.arcsin((markers.bottom_right.x - markers.top_right.x) / (markers.top_right.y - markers.bottom_right.y))
	gamma = np.arcsin((markers.bottom_right.y - markers.bottom_left.y) / (markers.bottom_right.x - markers.bottom_left.x))
	delta = np.arcsin((markers.bottom_left.x - markers.top_left.x) / (markers.top_left.y - markers.bottom_left.y))
	return (alpha + beta + gamma + delta) / 4  # in radians


def rotate_marker(marker: Marker, angle: float, image_shape: tuple[int, int]) -> tuple[int, int]:
	center_x = int(image_shape[0] / 2)
	center_y = int(image_shape[1] / 2)

	# Translate the point to the origin
	translated_point = (marker.center[0] - center_x, marker.center[1] - center_y)

	# Construct the rotation matrix
	rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
								[math.sin(angle), math.cos(angle)]])

	# Rotate the translated point
	rotated_point = np.dot(rotation_matrix, translated_point)

	# Translate the rotated point back to its original position
	rotated_point = (int(rotated_point[0] + center_x), int(rotated_point[1] + center_y))

	return rotated_point
