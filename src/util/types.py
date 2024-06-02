from typing import TypedDict

BoundingBox = tuple[int, int, int, int]


class PredictedEntity(TypedDict):
	label: str | None
	text: str
	boundingBox: BoundingBox
	ocr_confidence: float
