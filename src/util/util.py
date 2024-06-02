from difflib import SequenceMatcher


def find_best_label(target: str, entities: list[dict[str, str]]) -> str:
	similarities = map(lambda s: (list(s.values())[0], SequenceMatcher(None, target, list(s.keys())[0]).ratio()), entities)
	best_match = max(list(similarities), key=lambda x: x[1])
	return best_match[0]
