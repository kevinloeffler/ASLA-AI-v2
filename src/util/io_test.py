import os
import pytest

from src.util.io import get_model_path


@pytest.fixture
def temp_dir(tmp_path):
	return tmp_path


def test_create_new_directory(temp_dir):
	parent_dir = str(temp_dir / 'new_dir')
	model_name = 'model'
	expected_path = os.path.join(parent_dir, model_name)

	result = get_model_path(parent_dir, model_name)

	assert result == expected_path
	assert os.path.exists(parent_dir)
	assert not os.path.exists(expected_path)


def test_model_path_in_existing_directory(temp_dir):
	parent_dir = str(temp_dir)
	model_name = 'model'
	expected_path = os.path.join(parent_dir, model_name)

	result = get_model_path(parent_dir, model_name)

	assert result == expected_path


def test_model_path_with_existing_model(temp_dir):
	parent_dir = str(temp_dir)
	model_name = 'model'
	os.makedirs(parent_dir, exist_ok=True)
	open(os.path.join(parent_dir, model_name), 'a').close()  # Create the model file

	expected_path = os.path.join(parent_dir, model_name + '_v1')

	result = get_model_path(parent_dir, model_name)

	assert result == expected_path


def test_model_path_with_existing_versioned_models(temp_dir):
	parent_dir = str(temp_dir)
	model_name = 'model'
	os.makedirs(parent_dir, exist_ok=True)
	open(os.path.join(parent_dir, model_name), 'a').close()  # Create the model file
	open(os.path.join(parent_dir, model_name + '_v1'), 'a').close()  # Create version 1
	open(os.path.join(parent_dir, model_name + '_v2'), 'a').close()  # Create version 2

	expected_path = os.path.join(parent_dir, model_name + '_v3')

	result = get_model_path(parent_dir, model_name)

	assert result == expected_path


def test_trailing_slash_handling(temp_dir):
	parent_dir = str(temp_dir) + '/'
	model_name = 'model'
	expected_path = os.path.join(parent_dir, model_name)

	result = get_model_path(parent_dir, model_name)

	assert result == expected_path


def test_create_new_directory_with_trailing_slash(temp_dir):
	parent_dir = str(temp_dir / 'new_dir') + '/'
	model_name = 'model'
	expected_path = os.path.join(parent_dir, model_name)

	result = get_model_path(parent_dir, model_name)

	assert result == expected_path
	assert os.path.exists(parent_dir)
	assert not os.path.exists(expected_path)
