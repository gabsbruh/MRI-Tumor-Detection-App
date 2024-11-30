from app.SingleDetection import SingleDetection
import numpy as np

def test_integration_single_detection(mocker):
    sd = SingleDetection(None)
    mocker.patch.object(sd, 'preprocess_image', return_value=np.ones((1, 224, 224, 3)))
    mock_model = mocker.Mock()
    mock_model.predict.return_value = [[0.1, 0.9]]
    sd.model = mock_model
    sd.original_image = np.ones((224, 224, 3), dtype=np.uint8)
    sd.detect_tumor()
    assert sd.result.text() == "DETECTED"