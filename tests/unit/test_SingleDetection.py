from app.SingleDetection import SingleDetection

def test_browse_for_image(mocker):
    sd = SingleDetection(None)
    mocker.patch('app.SingleDetection.QFileDialog.getOpenFileName', return_value=("test_image.jpg", ""))
    sd.browse_for_img()
    assert sd.img_path.endswith(".jpg")

def test_detect_tumor(mocker):
    sd = SingleDetection(None)
    mocker.patch.object(sd, 'preprocess_image', return_value=np.ones((1, 224, 224, 3)))
    mock_model = mocker.Mock()
    mock_model.predict.return_value = [[0.1, 0.9]]
    sd.model = mock_model
    sd.original_image = np.ones((224, 224, 3), dtype=np.uint8)
    sd.detect_tumor()
    assert sd.result.text() == "DETECTED"