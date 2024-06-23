import pytest

from src.postprocessing.ai_postprocessing import postprocess, sort_and_flatten, entities_cleanup, convert_entities


@pytest.fixture
def predicted_entities():
	return [
		[
			{'label': 'LOC', 'text': 'wettingen', 'boundingBox': (100, 500, 200, 200), 'ocr_confidence': 0.85},
			{'label': 'CLT', 'text': 'Müller', 'boundingBox': (100, 400, 200, 200), 'ocr_confidence': 0.85},
			{'label': 'LOC', 'text': 'Zürich', 'boundingBox': (100, 350, 200, 200), 'ocr_confidence': 0.8},
			{'label': 'O', 'text': 'undefined', 'boundingBox': (100, 300, 100, 200), 'ocr_confidence': 0.75},
			{'label': 'DATE', 'text': '12.01.1922', 'boundingBox': (100, 200, 200, 200), 'ocr_confidence': 0.95},
			{'label': 'MST', 'text': '1:100', 'boundingBox': (100, 100, 200, 200), 'ocr_confidence': 0.8},
		],
		[
			{'label': 'MST', 'text': '1;100', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
		]
	]


@pytest.fixture
def artefacts():
	return ['Mueller', 'Zürich', 'Rapperswil']


def test_postprocess(predicted_entities, artefacts):
	result = postprocess(predicted_entities, artefacts)
	assert isinstance(result, list)
	assert all(isinstance(e, dict) for e in result)
	assert all('label' in e for e in result)
	assert all('text' in e for e in result)
	assert all('hasBoundingBox' in e for e in result)
	assert all('boundingBox' in e for e in result)
	assert all('manuallyChanged' in e for e in result)


def test_sorting(predicted_entities):
	result = sort_and_flatten(predicted_entities)
	assert result[0] == {'label': 'MST', 'text': '1;100', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85}
	assert result[-1] == {'label': 'LOC', 'text': 'wettingen', 'boundingBox': (100, 500, 200, 200), 'ocr_confidence': 0.85}


def test_filter_artefacts(predicted_entities, artefacts):
	result = postprocess(predicted_entities, artefacts)
	assert len(result) == 5


def test_scale_formatting():
	entities = [
		{'label': 'MST', 'text': '1;100', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
		{'label': 'MST', 'text': '1=100', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
		{'label': 'MST', 'text': 'M 1:100', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
		{'label': 'MST', 'text': 'Mst 1;100', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
	]
	assert postprocess([[entities[0]]], [])[0]['text'] == '1:100'
	assert postprocess([[entities[1]]], [])[0]['text'] == '1:100'
	assert postprocess([[entities[2]]], [])[0]['text'] == '1:100'
	assert postprocess([[entities[3]]], [])[0]['text'] == '1:100'


def test_date_formatting():
	entities = [
		{'label': 'DATE', 'text': '1.1.1950', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
		{'label': 'DATE', 'text': '01.01.1950', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
		{'label': 'DATE', 'text': 'Jan. 1950', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
		{'label': 'DATE', 'text': 'Januar 1950', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
		{'label': 'DATE', 'text': 'Jan 50', 'boundingBox': (100, 50, 200, 200), 'ocr_confidence': 0.85},
	]

	assert postprocess([[entities[0]]], [])[0]['text'] == '01.01.1950'
	assert postprocess([[entities[1]]], [])[0]['text'] == '01.01.1950'
	assert postprocess([[entities[2]]], [])[0]['text'] == '01.01.1950'
	assert postprocess([[entities[3]]], [])[0]['text'] == '01.01.1950'
	assert postprocess([[entities[4]]], [])[0]['text'] == '01.01.1950'


def test_convert_entities(predicted_entities):
	flat_entities = sort_and_flatten(predicted_entities)
	cleaned_entities = entities_cleanup(flat_entities)
	result = convert_entities(cleaned_entities)
	assert isinstance(result, list)
	assert len(result) == 5
	for entity in result:
		assert 'label' in entity
		assert 'text' in entity
		assert 'hasBoundingBox' in entity
		assert 'boundingBox' in entity
		assert 'manuallyChanged' in entity
