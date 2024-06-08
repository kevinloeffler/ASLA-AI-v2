import os
import random
from datetime import datetime


def _get_images_with_creation_dates(directory: str) -> list[tuple[str, float]]:
	images_with_dates = []
	for filename in os.listdir(directory):
		if filename.lower().endswith(('.jpg', '.jpeg')):
			filepath = os.path.join(directory, filename)
			creation_date = os.path.getctime(filepath)
			images_with_dates.append((filename, creation_date))
	return images_with_dates


def _calculate_weights(images_with_dates: list[tuple[str, float]]) -> list[float]:
	current_time = datetime.now().timestamp()
	weights = []
	for _, creation_date in images_with_dates:
		weight = current_time - creation_date
		weights.append(weight)
	return weights


def _weighted_random_choices(images_with_dates: list[tuple[str, float]], weights: list[float], count: int) -> list[str]:
	total_weight = sum(weights)
	probabilities = [weight / total_weight for weight in weights]

	# if there are not enough training images, return all
	if count >= len(images_with_dates):
		return [image[0] for image in images_with_dates]

	chosen_images = set()
	while len(chosen_images) < count:
		chosen_image = random.choices(images_with_dates, probabilities)[0][0]
		chosen_images.add(chosen_image)
	return list(chosen_images)


def select_training_data(directory: str, count: int) -> list[tuple[str, str]]:
	images_with_dates = _get_images_with_creation_dates(directory)
	if not images_with_dates:
		raise ValueError(f'No images found for {directory}')

	weights = _calculate_weights(images_with_dates)
	chosen_images = _weighted_random_choices(images_with_dates, weights, count)
	return [(image, image.replace('.jpg', '.json')) for image in chosen_images]


'''
if __name__ == "__main__":
	directory = "../../data/test_images"
	data = select_training_data(directory, 5)
	print(data)
'''
