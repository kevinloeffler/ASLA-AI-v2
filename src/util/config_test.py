import json
import pytest
import os
from tempfile import NamedTemporaryFile

from src.util.config import update_model_config

# Sample configuration to be used in tests
sample_config = {
    "settings": {
        "retraining_count": 100,
        "retraining_iterations": 1000
    },
    "models": {
        "layout": {
            "name": "microsoft/layoutlmv3-large",
            "score": 0.267
        },
        "ocr": {
            "name": "microsoft/trocr-large-handwritten",
            "score": 0.77
        },
        "ner": {
            "name": "../models/v1",
            "score": 0.325
        }
    }
}

@pytest.fixture
def config_file():
    with NamedTemporaryFile(delete=False, mode='w') as temp_file:
        json.dump(sample_config, temp_file)
        temp_file_path = temp_file.name
    yield temp_file_path
    os.remove(temp_file_path)


def test_update_model_name(config_file):
    update_model_config(model_type='layout', name='new/layoutlmv3-large', path=config_file)
    with open(config_file, 'r') as file:
        config = json.load(file)
    assert config['models']['layout']['name'] == 'new/layoutlmv3-large'
    assert config['models']['layout']['score'] == 0.267


def test_update_model_score(config_file):
    update_model_config(model_type='ocr', score=0.85, path=config_file)
    with open(config_file, 'r') as file:
        config = json.load(file)
    assert config['models']['ocr']['name'] == 'microsoft/trocr-large-handwritten'
    assert config['models']['ocr']['score'] == 0.85


def test_update_model_name_and_score(config_file):
    update_model_config(model_type='ner', name='new/models/v2', score=0.5, path=config_file)
    with open(config_file, 'r') as file:
        config = json.load(file)
    assert config['models']['ner']['name'] == 'new/models/v2'
    assert config['models']['ner']['score'] == 0.5


def test_invalid_model_type(config_file):
    with pytest.raises(ValueError, match="Model type 'invalid_model' not found in configuration."):
        update_model_config(model_type='invalid_model', name='new/model', score=0.9, path=config_file)
