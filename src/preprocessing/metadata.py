import json


def create_metadata(
		entities: list,
		contrast: float = 1.0,
		brightness: float = 1.0,
		sharpness: float = 1.0,
		white_balance=None,
		crop_top: int = 0,
		crop_right: int = 0,
		crop_bottom: int = 0,
		crop_left: int = 0,
		rotation: float = 1.0,
) -> dict:
	metadata = {
		'entities': entities,
		'grading': {
			'contrast': contrast,
			'brightness': brightness,
			'sharpness': sharpness,
			'whiteBalance': white_balance or [1.0, 1.0, 1.0],
			'manuallyChanged': False,
		},
		'format': {
			'crop': {
				'top': crop_top,
				'right': crop_right,
				'bottom': crop_bottom,
				'left': crop_left,
			},
			'rotation': rotation,
			'manuallyChanged': False,
		}
	}
	return metadata


def read_metadata(path: str) -> dict:
	# TODO: error handling
	with open(path, "r") as file:
		data = json.load(file)

		if 'entities' not in data:
			raise ValueError("No entities found in metadata")

		return data

