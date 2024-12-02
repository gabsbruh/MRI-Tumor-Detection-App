import pytest
from app.BaseDetection import BaseDetection
import numpy as np
from PyQt5.QtWidgets import QApplication

app = QApplication([])

def test_switch_to_default_model():
    """Test switching to the default model."""
    base_detection = BaseDetection(None)
    base_detection.load_model_()
    assert base_detection.model is not None
    assert base_detection.model_is_default is True

def test_switch_to_custom_model(mocker):
    """Test switching to a custom model."""
    mocker.patch('app.BaseDetection.QFileDialog.getOpenFileName', return_value=("models/VGG_16.keras", ""))
    base_detection = BaseDetection(None)
    base_detection.load_model_(default_model=False)
    assert base_detection.model is not None
    assert base_detection.model_is_default is False