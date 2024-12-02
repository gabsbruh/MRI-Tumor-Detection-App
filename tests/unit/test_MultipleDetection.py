from app.MultipleDetection import MultipleDetection
from PyQt5.QtWidgets import QApplication

app = QApplication([])

def test_browse_for_images(mocker):
    md = MultipleDetection(None)
    mocker.patch('app.MultipleDetection.QFileDialog.getExistingDirectory', return_value="../../archive/test/")
    mocker.patch('os.listdir', return_value=["image1.jpg", "image2.jpg"])
    md.browse_for_img()
    assert len(md.filenames) == 2

def test_export_to_csv(mocker):
    md = MultipleDetection(None)
    mocker.patch('app.MultipleDetection.QFileDialog.getSaveFileName', return_value=("output.csv", ""))
    md.table_model.setItem(0, 0, mocker.Mock(text="image1.jpg"))
    md.table_model.setItem(0, 1, mocker.Mock(text="No Tumor"))
    md.export_to_csv()