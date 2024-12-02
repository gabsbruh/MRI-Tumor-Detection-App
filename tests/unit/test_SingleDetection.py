from app.SingleDetection import SingleDetection
from PyQt5.QtWidgets import QApplication
import numpy as np
from unittest import mock

app = QApplication([])

def test_browse_for_image(mocker):
    sd = SingleDetection(None)
    mocker.patch('app.SingleDetection.QFileDialog.getOpenFileName', return_value=("archive/test/Y_new.png", ""))
    sd.browse_for_img()
    assert sd.img_path.endswith("Y_new.png")

def test_detect_tumor(mocker):
    sd = SingleDetection(None)
    mocker.patch.object(sd, 'preprocess_image', return_value=np.ones((1, 224, 224, 3)))
    mock_model = mock.Mock()
    mock_model.predict.return_value = [[0.1, 0.9]]
    sd.model = mock_model
    sd.original_image = np.ones((224, 224, 3), dtype=np.uint8)
    sd.detect_tumor()
    assert sd.result.text() == "DETECTED"
