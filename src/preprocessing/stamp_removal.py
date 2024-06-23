import os

import cv2
import numpy as np


def remove_stamp(image: np.ndarray, stamp_path: str = 'src/preprocessing/asla_stamp.jpg', certainty_threshold: float = 0.32):
	if not os.path.isfile(stamp_path):
		raise FileNotFoundError(f'Could not open stamp file at: {stamp_path}')

	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	stamp = cv2.cvtColor(cv2.imread(stamp_path), cv2.COLOR_BGR2GRAY)

	margin = 10  # add a small margin to avoid black lines
	y_margin = 60  # needed to capture the whole stamp (the last line ist cropped to increase matches)

	result = cv2.matchTemplate(gray_image, stamp, cv2.TM_CCOEFF_NORMED)
	_, max_value, min_loc, max_loc = cv2.minMaxLoc(result)

	if max_value < certainty_threshold:
		print(f'No stamp found on image')
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	start_x, start_y = max_loc
	end_x = start_x + stamp.shape[1]
	end_y = start_y + stamp.shape[0] + y_margin

	fill_color = __get_image_background(image, (start_x, start_y), (end_x, end_y))
	cv2.rectangle(image, (start_x - margin, start_y - margin), (end_x + margin, end_y), color=fill_color, thickness=-1)

	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def __get_image_background(image, top_left: tuple[int, int], bottom_right: tuple[int, int], margin: int = 10) -> tuple[int, int, int]:
	try:
		# get different samples of the background color
		sample_1 = image[top_left[1] - margin, top_left[0] - margin]
		sample_2 = image[top_left[1] - margin, top_left[0]]
		sample_3 = image[top_left[1], top_left[0] - margin]
		sample_4 = image[bottom_right[1] + margin, bottom_right[0] + margin]
		sample_5 = image[bottom_right[1] + margin, bottom_right[0]]
		sample_6 = image[bottom_right[1], bottom_right[0] + margin]
		average_values = tuple(sum(x) // len(x) for x in zip(sample_1, sample_2, sample_3, sample_4, sample_5, sample_6))
		return int(average_values[0]), int(average_values[1]), int(average_values[2])
	except Exception as error:
		print(error)
		return 255, 255, 255
