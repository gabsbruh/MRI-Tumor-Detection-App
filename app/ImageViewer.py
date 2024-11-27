from PyQt5.Qt import Qt
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtGui import QIcon, QPixmap


class ImageViewer(QMainWindow):
    def __init__(self, full_img_path):
        super().__init__()
        
        self.full_img_path = full_img_path
        filename = full_img_path.split('/')[-1]
        # window settings
        self.setWindowTitle(f"{filename}")
        self.setWindowIcon(QIcon("static/img/icon.png"))
        self.setProperty("class", "default_widget")
        
        # load stylesheet
        self.load_stylesheet(path="static/css/styles.css")
        
        label = QLabel(alignment=Qt.AlignCenter)
        pixmap = QPixmap(self.full_img_path)
        max_width = 500
        max_height = 500 
        scaled_pixmap = pixmap.scaled(max_width, max_height, aspectRatioMode=Qt.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)
        
        # window size matched to img size
        self.setCentralWidget(label)
        self.resize(scaled_pixmap.width(), scaled_pixmap.height())

        self.show()  # Ensure the show method is properly called
    
    def load_stylesheet(self, path):
        with open(path, 'r') as f:
            styles = f.read()
            self.setStyleSheet(styles)
