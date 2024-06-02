from difflib import SequenceMatcher

from src.ai.ner.ner_model import NerModel


def ner_benchmark(ner_model: NerModel, sentence: str, target_entities: list[dict[str, any]]) -> float:
	true_pos = 0
	false_pos = 0

	predictions: list[dict[str, str]] = ner_model.predict(sentence)
	filtered_predictions = list(filter(lambda p: list(p.values())[0] != 'O', predictions))
	print(f'filtered predictions: {filtered_predictions}')
	for prediction in filtered_predictions:
		for entity in target_entities:
			if __compare(prediction, entity):
				true_pos += 1
				false_pos -= 1
				break
		false_pos += 1

	false_neg = len(target_entities) - true_pos

	print(f'true_pos: {true_pos}, false_pos: {false_pos}, false_neg: {false_neg}')

	if true_pos + false_pos == 0:
		precision = 0
	else:
		precision = true_pos / (true_pos + false_pos)

	if true_pos + false_neg == 0:
		recall = 0
	else:
		recall = true_pos / (true_pos + false_neg)

	if precision + recall == 0:
		return 0
	return 2 * precision * recall / (precision + recall)


def __compare(prediction: dict[str, str], target_entity: dict[str, any]) -> bool:
	# print(f'comparing: "{list(prediction.values())[0]}" and {target_entity["label"]}')
	if list(prediction.values())[0].lower() == target_entity['label'].lower():
		text_similarity = SequenceMatcher(None, list(prediction.keys())[0], target_entity['text']).ratio()
		return text_similarity > 0.7
	return False


if __name__ == '__main__':
	model = NerModel('bert', '../../../models/v1', 1)
	f1_score = ner_benchmark(model, 'Badeanlage in Aarau, M 1:100, 23.9.48', [
		{
			"label": "LOC",
			"text": "Aarau",
			"hasBoundingBox": True,
			"boundingBox": {
				"top": 940,
				"right": 2571,
				"bottom": 996,
				"left": 2427
			},
			"manuallyChanged": True
		},
		{
			"label": "MST",
			"text": "M 1:100",
			"hasBoundingBox": True,
			"boundingBox": {
				"top": 996,
				"right": 2860,
				"bottom": 1050,
				"left": 2710
			},
			"manuallyChanged": True
		},
		{
			"label": "DATE",
			"text": "23.9.48",
			"hasBoundingBox": True,
			"boundingBox": {
				"top": 1045,
				"right": 3222,
				"bottom": 1093,
				"left": 3037
			},
			"manuallyChanged": True
		}
	])
	print('F1 score:', f1_score)
