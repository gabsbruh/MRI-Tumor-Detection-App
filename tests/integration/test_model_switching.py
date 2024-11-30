import pytest
from app.BaseDetection import BaseDetection
import numpy as np

def test_switch_to_default_model():
    """Test switching to the default model."""
    base_detection = BaseDetection(None)
    base_detection.load_model_()
    assert base_detection.model is not None
    assert base_detection.model_is_default is True

def test_switch_to_custom_model(mocker):
    """Test switching to a custom model."""
    mocker.patch('app.BaseDetection.QFileDialog.getOpenFileName', return_value=("custom_model.keras", ""))
    base_detection = BaseDetection(None)
    base_detection.load_model_(default_model=False)
    assert base_detection.model is not None
    assert base_detection.model_is_default is False

def test_preprocessing_after_model_switch(mocker):
    """Ensure the correct preprocessing is applied after switching models."""
    base_detection = BaseDetection(None)
    
    # Load the default model
    base_detection.load_model_()
    default_image = np.ones((256, 256, 3), dtype=np.uint8)
    processed_default = base_detection.preprocess_image(default_image)
    assert processed_default.shape == (1, 224, 224, 3)  # VGG16 preprocessing shape

    # Switch to custom model
    mocker.patch('app.BaseDetection.QFileDialog.getOpenFileName', return_value=("custom_model.keras", ""))
    base_detection.load_model_(default_model=False)
    custom_image = np.ones((256, 256, 3), dtype=np.uint8)
    processed_custom = base_detection.preprocess_image(custom_image)
    assert processed_custom.shape[1:] == (224, 224, 3)  # Ensure custom preprocessing is used