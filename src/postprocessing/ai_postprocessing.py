import re
from difflib import SequenceMatcher
from dateutil import parser

from src.util.types import PredictedEntity


def postprocess(predicted_entities: list[list[PredictedEntity]], artefacts: list[str]) -> list:
	entities = sort_and_flatten(predicted_entities)
	entities = filter_artefacts(entities, artefacts)
	entities = entities_cleanup(entities)
	return convert_entities(entities)


def sort_and_flatten(entity_clusters: list[list[PredictedEntity]]) -> list[PredictedEntity]:
	entities = [e for ents in entity_clusters for e in ents]
	entities = filter(lambda e: e['label'] != 'O', entities)
	return sorted(entities, key=lambda entity: entity['boundingBox'][1])


def filter_artefacts(entities: list[PredictedEntity], artefacts: list[str], similarity_threshold=0.2) -> list[
	PredictedEntity]:
	filtered_entities = []
	for entity in entities:
		if not any(SequenceMatcher(None, entity['text'], artefact).ratio() > similarity_threshold for artefact in artefacts):
			filtered_entities.append(entity)
	return filtered_entities


def entities_cleanup(entities: list[PredictedEntity]) -> list[PredictedEntity]:
	entities = format_scale(entities)
	entities = format_date(entities)
	# Optional: Filter based on ocr score if many results per label?
	return entities


def format_scale(entities: list[PredictedEntity]) -> list[PredictedEntity]:
	formatted_entities = []
	scale_pattern = re.compile(r'\d+:\d+')
	delimiter_pattern = re.compile(r'\s*[:;.,=]\s*')

	for entity in entities:
		if entity['label'] == 'MST':
			text = re.sub(delimiter_pattern, ':', entity['text'])
			match = re.search(scale_pattern, text)
			if match:
				entity['text'] = match.group()
				formatted_entities.append(entity)
		else:
			formatted_entities.append(entity)
	return formatted_entities


def format_date(entities: list[PredictedEntity]) -> list[PredictedEntity]:
	formatted_entities = entities
	for entity in formatted_entities:
		if entity['label'] == 'DATE':
			date_str = entity['text']

			date_str = re.sub(r'[\.,]', ' ', date_str)
			date_str = re.sub(r'\s+', ' ', date_str).strip()
			date_str = re.sub(r'\bJan\w*', 'January', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bFeb\w*', 'February', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bMar\w*', 'March', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bApr\w*', 'April', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bJun\w*', 'June', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bJul\w*', 'July', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bAug\w*', 'August', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bSep\w*', 'September', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bOct\w*', 'October', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bNov\w*', 'November', date_str, flags=re.IGNORECASE)
			date_str = re.sub(r'\bDec\w*', 'December', date_str, flags=re.IGNORECASE)

			try:
				month_year_match = re.match(
					r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}$',
					date_str, flags=re.IGNORECASE)
				if month_year_match:
					date_str = '01 ' + date_str

				parsed_date = parser.parse(date_str, dayfirst=True, ignoretz=True, fuzzy=True)
				if parsed_date.year > 2000:
					new_year = parsed_date.year - 100
					parsed_date = parsed_date.replace(year=new_year)
				new_date = parsed_date.strftime('%d.%m.%Y')
				entity['text'] = new_date
			except ValueError:
				continue
	return formatted_entities


def convert_entities(predicted_entities: list[PredictedEntity]) -> list:
	"""Convert predicted entities to metadata entities"""
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
