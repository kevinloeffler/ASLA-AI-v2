from .data_service import select_training_data
from ..ai.ai import test_models
from ..ai.ner.ner_training import train_ner_model
from ..util.config import load_config, update_model_config


def run_retraining():
	config = load_config()

	training_data = select_training_data('../data/retraining/', count=config.get('settings', {}).get('retraining_count', 10))

	# Run retraining for each model: extract training data & retrain
	# TODO: retrain_layout_model()
	# TODO: retrain_ocr_model()

	retrained_ner_model = train_ner_model(
		data_dir='../data/retraining/',
		data=training_data,
		model_name=config['models']['ner']['name'],
		model_type='bert',
		iterations=config.get('settings', {}).get('retraining_iterations', 10),
		safe_to_dir='../models/ner/'
	)

	# Run model test (without safe to disk)
	score = test_models(
		layout_model_name=config['models']['layout']['name'],
		ocr_model_name=config['models']['ocr']['name'],
		ner_model_name=retrained_ner_model,
		ner_model_type='bert',
		directory='../data/test_images/',
		output_dir='../benchmarks',
		safe=False
	)

	# Compare score, if better -> replace
	# TODO: if score['layout_model'] > config['models']['layout']['score']:
	# TODO: if score['ocr_model'] > config['models']['ocr']['score']:

	if score['ner_model'] > config['models']['ner']['score']:
		update_model_config(model_type='ner', name=retrained_ner_model, score=score['ner_model'])

	# Update test set

	return
