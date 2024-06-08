import os
import random

import cv2
from datasets import load_metric, Metric

from .ocr_model import OcrModel
from preprocessing.metadata import read_metadata

import torch
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, TrOCRProcessor, Seq2SeqTrainingArguments, default_data_collator

# from ...util.types import BoundingBox
BoundingBox = any

def _load_entities(from_directory: str, split=0.2) -> tuple[list[tuple[str, str, BoundingBox]], list[tuple[str, str, BoundingBox]]]:
	data: list[tuple[str, str, BoundingBox]] = []
	for file in os.listdir(from_directory):
		if not file.endswith(".json"):
			continue
		image_name = file.replace('.json', '.jpg')
		if not os.path.exists(os.path.join(from_directory, image_name)):
			continue
		try:
			metadata = read_metadata(os.path.join(from_directory, file))
		except Exception:
			continue

		for entity in metadata['entities']:
			data.append((os.path.join(from_directory, image_name), entity['text'], entity['boundingBox']))

	random.shuffle(data)
	split_point = int(len(data) * split)

	return data[split_point:], data[:split_point]  # train, test


class OcrTrainingDataset(Dataset):
	def __init__(self, dataset: list[tuple[str, str, BoundingBox]], processor: TrOCRProcessor, max_target_length=128):
		self.dataset = dataset
		self.processor = processor
		self.max_target_length = max_target_length

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		path: str = self.dataset[index][0]
		text: str = self.dataset[index][1]
		bounding_box: dict = self.dataset[index][2]

		image = cv2.imread(path)
		cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		cropped_image = image[bounding_box['top']: bounding_box['bottom'], bounding_box['left']: bounding_box['right']]
		pixel_values = self.processor(cropped_image, return_tensors='pt').pixel_values

		labels = self.processor.tokenizer(text, padding='max_length', max_length=self.max_target_length).input_ids
		labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]  # Ignore pad tokens in the loss function

		return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}


def _compute_metrics(prediction, metric: Metric, processor: TrOCRProcessor):
	labels_ids = prediction.label_ids
	pred_ids = prediction.predictions

	pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
	labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
	label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

	cer = metric.compute(predictions=pred_str, references=label_str)
	return {"cer": cer}


def train_ocr(base_model: str, data_directory: str, output_directory: str):
	processor = TrOCRProcessor.from_pretrained(base_model)

	train_data, test_data = _load_entities(from_directory=data_directory)
	train = OcrTrainingDataset(dataset=train_data, processor=processor)
	test = OcrTrainingDataset(dataset=test_data, processor=processor)

	model = OcrModel(base_model).model
	model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
	model.config.pad_token_id = processor.tokenizer.pad_token_id
	model.config.vocab_size = model.config.decoder.vocab_size
	model.config.eos_token_id = processor.tokenizer.sep_token_id
	model.config.max_length = 64
	model.config.early_stopping = True
	model.config.no_repeat_ngram_size = 3
	model.config.length_penalty = 2.0
	model.config.num_beams = 4

	training_args = Seq2SeqTrainingArguments(
		predict_with_generate=True,
		evaluation_strategy="steps",
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		fp16=True,
		output_dir=output_directory,
		logging_steps=10,
		save_steps=1000,
		eval_steps=200,
		save_total_limit=2,
		dataloader_num_workers=4,
		gradient_accumulation_steps=2,
	)

	cer_metric = load_metric("cer")

	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		compute_metrics=lambda pred: _compute_metrics(pred, cer_metric, processor),
		train_dataset=train,
		eval_dataset=test,
		data_collator=default_data_collator,
	)
	trainer.train()

	trainer.save_model(output_directory)
	processor.save_pretrained(output_directory)
