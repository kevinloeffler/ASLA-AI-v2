import os


def get_model_path(parent_dir: str, model_name: str) -> str:
	directory = parent_dir if parent_dir.endswith('/') else parent_dir + '/'

	if not os.path.exists(directory):
		print('IO: created directories:', directory)
		os.makedirs(directory)
		return directory + model_name

	if not os.path.exists(directory + model_name):
		return directory + model_name

	suffix = 1

	while os.path.exists(directory + model_name + f'_v{suffix}'):
		suffix += 1

	return directory + model_name + f'_v{suffix}'
