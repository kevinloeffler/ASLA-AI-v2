import os
import random

import torch
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs

from ...preprocessing.metadata import read_metadata
from ...util.io import get_model_path


def _augment_ner_entity(entity: dict[str, str]) -> str | None:
    text = entity["text"]

    if entity['label'].upper() == 'LOC':
        if random.random() < 0.7:
            return text
        templates = [
            'Dreihäuser Gruppe in $LOC', 'Überbauung $LOC', 'Blick nach dem Wohnhaus vom Eingang - $LOC',
            'Unser Garten $LOC', 'Kantonsspital $LOC', 'Schulhaus $LOC', 'Wohnhaus $LOC', 'Wohnsitz $LOC',
            'Kindergarten $LOC', 'Dachgarten in $LOC', 'Situation in $LOC', 'Hortensien Anlage $LOC',
            'Rasenfläche Neugestaltung Abtei $LOC', 'Garten in $LOC', 'Projekt neubau in $LOC'
        ]
        return random.choice(templates).replace('$LOC', text)

    elif entity['label'].upper() == 'CLT':
        if random.random() < 0.7:
            return text
        templates = [
            'Gartenhaus $CLT', 'Garten Herrn $CLT', 'Garten Hrn. $CLT', 'Wohnhaus $CLT',
            'Idee zur Ausgestaltung Projekt $CLT', 'Wasser Becken $CLT Werkzeichnung', 'Werkzeichnung Weg $CLT',
            'Dachgarten Haus $CLT', 'Werkzeichnung Weg $CLT', 'Stützmauer $CLT', 'Buschwindrosen des Herrn. $CLT',
            'Blumenbeete Haus $CLT', 'Gartengestaltung der Familie $CLT', 'Projekt des Kmd. $CLT'
        ]
        return random.choice(templates).replace('$CLT', text)

    elif entity['label'].upper() == 'MST':
        if random.random() < 0.3:
            return text
        templates = ['Masstab $MST', 'Masstab: $MST', 'Mst: $MST', 'M. $MST', 'M: $MST', 'M. = $MST', 'M = $MST']
        return random.choice(templates).replace('$MST', text)

    elif entity['label'].upper() == 'DATE':
        return text

    return None


def _load_data(data_dir: str, data: list[tuple[str, str]]) -> pd.DataFrame:
    data = [read_metadata(data_dir + metadata[1]) for metadata in data]

    sentence_ids = []
    words = []
    labels = []

    for m_index, metadata in enumerate(data):
        entities: list[dict] = metadata['entities']
        for e_index, entity in enumerate(entities):
            sentence_id = m_index + e_index
            sentence = _augment_ner_entity(entity)
            for word in sentence.split(' '):
                sentence_ids.append(sentence_id)
                words.append(word)
                label = entity['label'].upper() if word in entity['text'].split(' ') else 'O'
                labels.append(label)

    conll_df = pd.DataFrame({
        'sentence_id': sentence_ids,
        'words': words,
        'labels': labels
    })

    return conll_df


def train_ner_model(
        data_dir: str,
        data: list[tuple[str, str]],
        model_type: str,
        model_name: str,
        iterations: int,
        safe_to_dir: str = None,
) -> str:
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'mps' if torch.backends.mps.is_available() else 'cpu')

    labels = ['O', 'CLT', 'LOC', 'MST', 'CLOC', 'DATE']

    safe_to = safe_to_dir if safe_to_dir is not None else '/'.join(model_name.split('/')[:-1])
    safe_to_path = get_model_path(safe_to, 'ner')

    model_args = NERArgs()
    model_args.device = device
    model_args.labels_list = labels
    model_args.num_train_epochs = iterations
    model_args.classification_report = True
    model_args.use_multiprocessing = True
    model_args.save_model_every_epoch = False
    model_args.output_dir = safe_to_path

    model = NERModel(model_type, model_name, use_cuda=use_cuda, args=model_args)
    data = _load_data(data_dir, data)
    model.train_model(train_data=data, show_running_loss=True)

    return safe_to_path
