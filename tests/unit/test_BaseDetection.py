import pytest
from app.BaseDetection import BaseDetection
import numpy as np

def test_load_default_model():
    bd = BaseDetection(None)
    bd.load_model_()
    assert bd.model is not None
    assert bd.model_is_default

def test_load_custom_model(mocker):
    mocker.patch('app.BaseDetection.QFileDialog.getOpenFileName', return_value=("custom_model.keras", ""))
    bd = BaseDetection(None)
    bd.load_model_(default_model=False)
    assert bd.model is not None
    assert not bd.model_is_default

def test_preprocess_image():
    bd = BaseDetection(None)
    bd.load_model_()
    dummy_image = np.ones((256, 256, 3), dtype=np.uint8)
    processed_image = bd.preprocess_image(dummy_image)
    assert processed_image.shape == (1, 224, 224, 3)