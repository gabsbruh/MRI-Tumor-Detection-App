from app.MainWindow import MainWindow
from PyQt5.QtWidgets import QApplication

app = QApplication([])

def test_navigation():
    main_window = MainWindow()
    assert main_window.stacked_widget.currentWidget() == main_window.start_page
    main_window.show_page("SinglePage")
    assert main_window.stacked_widget.currentWidget() == main_window.single_page