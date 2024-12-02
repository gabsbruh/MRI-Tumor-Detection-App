from script.preprocess_input_custom import custom_preprocessing
import numpy as np
from PyQt5.QtWidgets import QApplication

app = QApplication([])

def test_custom_preprocessing():
    dummy_image = np.ones((256, 256, 3), dtype=np.uint8)
    processed_image = custom_preprocessing(dummy_image)
    assert processed_image.shape == (1, 224, 224, 3)