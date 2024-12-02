from app.MultipleDetection import MultipleDetection
from PyQt5.QtWidgets import QApplication
import numpy as np
app = QApplication([])

def test_integration_multiple_detection(mocker):
    md = MultipleDetection(None)
    mocker.patch('app.MultipleDetection.QFileDialog.getExistingDirectory', return_value="archive/test")
    mocker.patch.object(md, 'preprocess_image', return_value=np.ones((1, 224, 224, 3)))
    mock_model = mocker.Mock()
    mock_model.predict.side_effect = [[[0.1, 0.9]], [[0.9, 0.1]], [[0.1, 0.9]]]
    md.model = mock_model
    md.browse_for_img()
    md.detect_tumor()
    assert md.result.text() == "Detected: 2 / 3"