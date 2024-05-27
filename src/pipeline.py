import math
import cv2
import numpy as np

from preprocessing.format import format_image
from preprocessing.markers import Markers
from preprocessing.metadata import create_metadata
from ai.ai import extract_entities


def handle_image(raw_image: np.ndarray):
	# image
	image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # faster, use if color is not important
	image_height, image_width = image.shape[:2]

	# format
	markers = Markers(image_gray)
	crop, angle = format_image(markers=markers, image_shape=(image_width, image_height))

	# entity extraction
	entities = extract_entities(image)

	# create metadata
	metadata = create_metadata(
		entities=entities,
		contrast=1.0,
		brightness=1.0,
		sharpness=1.0,
		white_balance=[0.99, 1.0, 1.01],
		crop_top=crop[0],
		crop_right=crop[1],
		crop_bottom=crop[2],
		crop_left=crop[3],
		rotation=math.degrees(angle),
	)

	return metadata
