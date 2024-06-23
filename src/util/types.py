from typing import TypedDict

BoundingBox = tuple[int, int, int, int]  # left, top, right, bottom


class PredictedEntity(TypedDict):
	label: str | None
	text: str
	boundingBox: BoundingBox
	ocr_confidence: float
