from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QMessageBox
from PyQt5.QtGui import QIcon
from app.MultipleDetection import MultipleDetection
from app.StartPage import StartPage
from app.SingleDetection import SingleDetection

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # window settings
        self.setFixedSize(1000, 600)
        self.setWindowTitle("Brain Tumor Detection")
        self.setWindowIcon(QIcon("static/img/icon.png"))
        
        # Create stack of widgets
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Initialize widgets
        self.start_page = StartPage(self.show_page)
        self.multi_page = MultipleDetection(self.show_page)
        self.single_page = SingleDetection(self.show_page)
        
        # Add widgets
        self.stacked_widget.addWidget(self.start_page)
        self.stacked_widget.addWidget(self.multi_page)
        self.stacked_widget.addWidget(self.single_page)
        
        # set start page
        self.stacked_widget.setCurrentWidget(self.start_page)
        self.setProperty("class", "start_widget")
        
        # load stylesheet
        self.load_stylesheet(path="static/css/styles.css")
        
    def show_page(self, page_name):
        """Switch trough different QWidgets in main widget (QStackedWidget) by setting widget button reffering
        to with its class name (in variable page_name). Then change also css styling of background based on property name
        """
        if page_name == "Start":
            self.stacked_widget.setCurrentWidget(self.start_page)
            self.setProperty("class", "start_widget")
        elif page_name == "MultiPage":
            self.stacked_widget.setCurrentWidget(self.multi_page)
            self.setProperty("class", "default_widget")
        elif page_name == "SinglePage":
            self.stacked_widget.setCurrentWidget(self.single_page)
            self.setProperty("class", "default_widget")
        else:
            QMessageBox.warning(self, "Redirection Error", "An error during redirection occured.")
            return
        self.style().polish(self)
    
    def load_stylesheet(self, path):
        with open(path, 'r') as f:
            styles = f.read()
            self.setStyleSheet(styles)
    
