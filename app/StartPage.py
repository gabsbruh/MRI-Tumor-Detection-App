from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, 
    QPushButton, QSpacerItem
)


class StartPage(QWidget):
    def __init__(self, show_page_callback):
        super().__init__()
        self.show_page = show_page_callback
        # CSS identifier
        self.setObjectName("start-widget")
        
        # buttons
        app_name = QLabel("Brain Tumor Detection")
        app_name.setObjectName("TITLE")
        
        self.start_button = QPushButton("Single MRI")
        self.upload_model = QPushButton("Multiple MRI's")
        
        # layout
        self.master_layout = QVBoxLayout()
        
        # apply widgets
        self.master_layout.addSpacerItem(QSpacerItem(0, 120))
        self.master_layout.addWidget(app_name, alignment=Qt.AlignCenter)
        self.master_layout.addSpacerItem(QSpacerItem(0, 120))
        self.master_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        self.master_layout.addWidget(self.upload_model, alignment=Qt.AlignCenter)
        self.master_layout.addSpacerItem(QSpacerItem(0, 120))
        
        # front-end configurations
        self.start_button.setFixedWidth(500) # 500 is 50% of parent window width
        self.upload_model.setFixedWidth(500)
        self.start_button.setFixedHeight(50)
        self.upload_model.setFixedHeight(50)
        
        self.setLayout(self.master_layout)
        
        # connect buttons
        self.start_button.clicked.connect(lambda: self.show_page("SinglePage")) # route to image classification
        self.upload_model.clicked.connect(lambda: self.show_page("MultiPage")) # route to image classification
