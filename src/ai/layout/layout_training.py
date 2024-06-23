import json
import os

import cv2
from datasets import Dataset
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, DataCollatorForTokenClassification, \
	Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset as TorchDataset


def get_model_path(parent_dir: str, model_name: str) -> str:
	directory = parent_dir if parent_dir.endswith('/') else parent_dir + '/'

	if not os.path.exists(directory):
		print('IO: created directories:', directory)
		os.makedirs(directory)
		return directory + model_name

	if not os.path.exists(directory + model_name):
		return directory + model_name

	suffix = 1

	while os.path.exists(directory + model_name + f'_v{suffix}'):
		suffix += 1

	return directory + model_name + f'_v{suffix}'


id2label = {
	0: "O",
	1: "CLT",
	2: "LOC",
	3: "MST",
	4: "DATE",
}

label2id = {
	"O": 0,
	"CLT": 1,
	"LOC": 2,
	"MST": 3,
	"DATE": 4,
}


class LayoutDataset(TorchDataset):
	def __init__(self, hf_dataset, processor: LayoutLMv3Processor):
		self.dataset = hf_dataset
		self.processor = processor

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		item = self.dataset[idx]
		image = cv2.imread(item['image'])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		encoded_inputs = self.processor(
			images=image,
			text=item['tokens'],
			boxes=item['bboxes'],
			word_labels=[label2id[label] for label in item['ner_tags']],
			padding="max_length",
			truncation=True,
			return_tensors="pt",
		)
		for k, v in encoded_inputs.items():
			encoded_inputs[k] = v.squeeze()

		encoded_inputs['bbox'] = encoded_inputs['bbox'].to(torch.int64)

		return encoded_inputs


def read_image_and_annotations(image_path: str):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	with open(image_path.replace('.jpg', '.json'), 'r') as f:
		annotations = json.load(f)
	return image, annotations


def normalize_bboxes(image_shape: tuple[int, ...], bbox: dict):
	image_height = image_shape[0]
	image_width = image_shape[1]
	x1 = bbox['left'] / image_width
	y1 = bbox['top'] / image_height
	x2 = (image_width - bbox['right']) / image_width
	y2 = (image_height - bbox['bottom']) / image_height
	return [x1, y1, x2, y2]


def create_raw_dataset(from_directory: str) -> Dataset:
	raw_data = []
	for file in os.listdir(from_directory):
		filepath = os.path.join(from_directory, file)
		if not file.endswith('.jpg'):
			continue

		image, metadata = read_image_and_annotations(filepath)

		datapoint = {
			'tokens': [],
			'bboxes': [],
			'ner_tags': [],
			'image': filepath,
		}

		for entity in metadata['entities']:
			datapoint['tokens'].append(entity['text'])
			datapoint['bboxes'].append(normalize_bboxes(image.shape, entity['boundingBox']))
			datapoint['ner_tags'].append(entity['label'].upper())

		raw_data.append(datapoint)

	return Dataset.from_list(raw_data)


def create_datasets(from_dataset: Dataset, processor: LayoutLMv3Processor, split=0.1) -> tuple[LayoutDataset, LayoutDataset]:
	dts = from_dataset.train_test_split(test_size=split)
	return LayoutDataset(dts['train'], processor), LayoutDataset(dts['test'], processor)


def train_layout_model(
		data_dir: str,
		# data: list[tuple[str, str]],
		base_model: str,
		iterations: int,
		safe_to_dir: str,
):
	processor = LayoutLMv3Processor.from_pretrained(base_model, apply_ocr=False)
	model = LayoutLMv3ForTokenClassification.from_pretrained(base_model, num_labels=len(label2id))

	raw_data = create_raw_dataset(data_dir)
	train, test = create_datasets(raw_data, processor)

	data_collator = DataCollatorForTokenClassification(processor.tokenizer)

	safe_to_path = get_model_path(safe_to_dir, 'layout')

	training_args = TrainingArguments(
		output_dir=safe_to_path,
		num_train_epochs=iterations,
		# learning_rate=2e-5,
		# weight_decay=0.01,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		warmup_steps=100,
		weight_decay=0.01,
		learning_rate=5e-5,
		evaluation_strategy="steps",
		gradient_accumulation_steps=2,
		eval_steps=500,
		save_steps=500,
		load_best_model_at_end=True,
		metric_for_best_model="accuracy",
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train,
		eval_dataset=test,
		data_collator=data_collator,
		tokenizer=processor.tokenizer,
	)

	trainer.train()

	model.save_pretrained(safe_to_path)
	processor.feature_extractor.save_pretrained(safe_to_path)


if __name__ == '__main__':
	train_layout_model(
		data_dir='../../../data/retraining/',
		base_model='microsoft/layoutlmv3-base',
		iterations=10,
		safe_to_dir="../../../models/layout/"
	)
