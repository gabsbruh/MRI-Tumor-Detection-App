from app.SingleDetection import SingleDetection
from PyQt5.QtWidgets import QApplication

app = QApplication([])

def test_ui_single_detection(mocker):
    single_detection = SingleDetection(None)
    mocker.patch('app.SingleDetection.QFileDialog.getOpenFileName', return_value=["archive/test/Y_new.png", ""])
    single_detection.browse_for_img()
    assert single_detection.path.text().endswith("archive/test/Y_new.png")