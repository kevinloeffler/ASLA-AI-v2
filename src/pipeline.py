import math
import cv2
import numpy as np

from .ai.ai import ModelWrapper
from .postprocessing.ai_postprocessing import postprocess
from .preprocessing.format import format_image
from .preprocessing.markers import Markers
from .preprocessing.metadata import create_metadata
from .preprocessing.image_processing import preprocess_image


def handle_image(raw_image: np.ndarray, ai: ModelWrapper, artefacts: list[str]) -> dict:
	# image
	image = cv2.imdecode(raw_image, cv2.IMREAD_COLOR)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # faster than color, use where possible
	image_height, image_width = image.shape[:2]

	# format
	markers = Markers(image_gray)
	crop, angle = format_image(markers=markers, image_shape=(image_width, image_height))

	# preprocess
	processed_image = preprocess_image(image)

	# entity extraction
	extracted_entities = ai.run_pipeline(processed_image)

	# postprocess
	entities = postprocess(extracted_entities, artefacts)

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
