from src.util.types import PredictedEntity


def postprocess(predicted_entities: list[list[PredictedEntity]]) -> list:
	clean_entities = entities_cleanup(predicted_entities)
	filtered_predicted_entities = filter_unlabeled_entities(clean_entities)
	return convert_entities(filtered_predicted_entities)


def entities_cleanup(entities: list[list[PredictedEntity]]) -> list[PredictedEntity]:
	# TODO:
	#  Format scales eg: 1:100
	#  Format Dates, how?
	#  Filter based on ocr score if many results per label
	return [e for ents in entities for e in ents]


def filter_unlabeled_entities(entities: list[PredictedEntity]) -> list[PredictedEntity]:
	return filter(lambda e: e['label'] != 'O', entities)


def convert_entities(predicted_entities: list[PredictedEntity]) -> list:
	"""Convert predicted entities to metadata entities and flatten clusters into a single list"""
	entities = []
	for predicted_entity in predicted_entities:
		entities.append({
			'label': predicted_entity['label'],
			'text': predicted_entity['text'],
			'hasBoundingBox': True,
			'boundingBox': {
				'left': predicted_entity['boundingBox'][0],
				'top': predicted_entity['boundingBox'][1],
				'right': predicted_entity['boundingBox'][2],
				'bottom': predicted_entity['boundingBox'][3]
			},
			'manuallyChanged': False,
		})
	return entities
