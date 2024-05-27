import cv2
from numpy import ndarray
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


class OcrModel:
	def __init__(self, model_name: str):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.processor = TrOCRProcessor.from_pretrained(model_name)
		self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
		# model config
		self.model.config.max_new_tokens = 10
		self.model.config.early_stopping = True
		self.model.config.no_repeat_ngram_size = 3
		self.model.config.length_penalty = 2.0
		self.model.config.num_beams = 4

	def predict(self, image: ndarray, confidence=-0.2) -> str or None:
		pixel_values = self.processor(image, return_tensors='pt').pixel_values.to(self.device)
		result = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
		generated_ids = result['sequences']
		score = result['sequences_scores'].numpy()[0]

		if score > confidence:
			prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
			return self.post_process_prediction(prediction)

	@staticmethod
	def post_process_prediction(string: str) -> str:
		forbidden_end_chars = ['.']
		new_string = string.lower()
		new_string = new_string[: -1] if new_string[-1] in forbidden_end_chars else new_string
		new_string = new_string.replace(' ', '')
		return new_string
