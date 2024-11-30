from app.SingleDetection import SingleDetection

def test_ui_single_detection(qtbot):
    single_detection = SingleDetection(None)
    qtbot.addWidget(single_detection)
    single_detection.browse_for_img()
    assert single_detection.path.text() == "/path/to/image.jpg"  # Mock path as needed