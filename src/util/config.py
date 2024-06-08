import json


def load_config(path='../ai_config.json') -> dict:
	with open(path) as file:
		return json.load(file)


def update_model_config(model_type: str, name: str = None, score: float = None, path='../ai_config.json'):
	with open(path, 'r') as file:
		config = json.load(file)

	new_name = name if name else config['models'][model_type]['name']
	new_score = score if score else config['models'][model_type]['score']

	if model_type in config['models']:
		config['models'][model_type]['name'] = new_name
		config['models'][model_type]['score'] = new_score
	else:
		raise ValueError(f"Model type '{model_type}' not found in configuration.")

	with open(path, 'w') as file:
		json.dump(config, file, indent=2)
