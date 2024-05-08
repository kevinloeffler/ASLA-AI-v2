from transformers import LayoutLMv3Processor


class ModelWrapper:

	layout_model = None
	ocr_model = None
	ner_model = None

	def __init__(self, layout_model: str, ocr_model: str, ner_model: str):
		self.layout_model = LayoutLMv3Processor.from_pretrained(layout_model)
		# self.ocr_model = TrOCR(ocr_model)
		# self.knn_model = Clustering()
		# self.ner_model = None
		print('Done loading models')


def extract_entities(image) -> list:

	return [
		{
			'label': 'loc',
			'text': 'WETTINGEN',
			'boundingBox': {
				'top': 399,
				'right': 3031,
				'bottom': 485,
				'left': 2532
			},
			'manuallyChanged': False
		},
		{
			'label': 'mst',
			'text': 'M. 1:100',
			'boundingBox': {
				'top': 407,
				'right': 3744,
				'bottom': 492,
				'left': 3395
			},
			'manuallyChanged': False
		},
		{
			'label': 'date',
			'text': '16.2.40',
			'boundingBox': {
				'top': 3545,
				'right': 2560,
				'bottom': 3616,
				'left': 2268
			},
			'manuallyChanged': False
		},
		{
			'label': 'date',
			'text': '1419',
			'boundingBox': {
				'top': 3609,
				'right': 2525,
				'bottom': 3680,
				'left': 2361
			},
			'manuallyChanged': False
		},
	]

