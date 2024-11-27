import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, 
    QLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from app.BaseDetection import BaseDetection

class SingleDetection(BaseDetection):
    def __init__(self, show_page_callback):
        super().__init__(show_page_callback)
        
        # no-ui instances
        self.img_path = ""
        self.original_image = None
        self.image_w_grad_original = None
        
        # unique widgets  
        self.result = QLabel("Detection Results", alignment=Qt.AlignCenter)
        self.result.setProperty("class", "tumor_detection")
        self.image = QLabel("Uploaded photo will be displayed here", alignment=Qt.AlignCenter)
             
        # main layout of this window
        self.master_layout = QVBoxLayout()
        self.hrows_layout = [QHBoxLayout() for _ in range(3)]
        
        # setup interface
        self.setup_interface()

          
    def setup_interface(self):        
        # setup first row
        self.hrows_layout[0].addWidget(self.go_back)
        self.hrows_layout[0].addWidget(self.browser)
        self.hrows_layout[0].addWidget(self.path)
        self.hrows_layout[0].addWidget(self.model_info)
        self.hrows_layout[0].addWidget(self.upload_model)
        
        # setup second row
        self.hrows_layout[1].addWidget(self.image)
        
        # setup third row
        self.hrows_layout[2].addWidget(self.grad_cam)
        self.hrows_layout[2].addWidget(self.save_grad)
        self.hrows_layout[2].addWidget(self.result)
        self.hrows_layout[2].addWidget(self.detect)
    
        # configure widgets position
        self.image.setFixedSize(1000, 480)
        
        # margins
        self.hrows_layout[0].contentsMargins()
    
        
        # add rows to master
        for row in self.hrows_layout:
            self.master_layout.addLayout(row)
        
        # set master
        self.setLayout(self.master_layout)

    def browse_for_img(self):
        """ Open file dialog to upload image. 
        Set image label to uploaded image.
        """
        self.img_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.svg *.bmp)"
        )
        if self.img_path:
            # set detection results to default
            self.result.setText("Detection Results")
            # set image and resize properly to display
            self.original_image = cv2.imread(self.img_path) # cv2.imread to load original image
            pixmap = QPixmap(self.img_path)
            if pixmap.isNull():
                QMessageBox.critical(self, "Image Error", "Image not loaded properly. Check extension")
                return None
            scale_pixmap = pixmap.scaledToHeight(480, Qt.SmoothTransformation)
            self.image.setPixmap(scale_pixmap)
            # set qlineedit
            self.path.setText(self.img_path)
            # set colors to default
            self.image.setObjectName("default")
            self.result.setObjectName("default")
            self.style().polish(self.result)
            self.style().polish(self.image)            
        else:
            QMessageBox.warning(self, "No File Chosen", "To load image, choose img file.")
            return None

    def detect_tumor(self):
        """Predict wheter on image is tumor or not. Sets detect qlabel based on the outcome.
        
        WARNING: if you choose custom model, it should contain custom_preprocessing function 
        in module script/preprocess_input_custom.py. Otherwise your model will take raw image,
        returning error (input type will be uint8 with shape:(xdim, ydim, 3))
        """
        if self.original_image is None:
            QMessageBox.warning(self, "Load Image", "Load image to use detect.")
            return None
        self.result.setText("")
        self.result.update()
        preprocessed_img = self.preprocess_image(self.original_image)
        output = self.model.predict(preprocessed_img)
        if np.argmax(output) == 1:
            self.result.setText("DETECTED")
            self.result.setObjectName("TUMOR_DETECTED")
            self.image.setObjectName("TUMOR_DETECTED_IMAGE")
        else:
            self.result.setText("No Tumor")
            self.result.setObjectName("NO_TUMOR")
            self.image.setObjectName("NO_TUMOR_IMAGE")
        # reload styling to apply to result
        self.style().polish(self.result)
        self.style().polish(self.image)
    
    def apply_grad_cam(self, checked):
        if checked:
            preprocessed_img = self.preprocess_image(self.original_image)
            heatmap = self.grad_cam_alg.compute_heatmap(preprocessed_img,  class_idx=0)
            self.image_w_grad_original = self.grad_cam_alg.overlay_heatmap(heatmap, preprocessed_img)
            repreprocessed_img = self.preprocess_image(self.image_w_grad_original, reversed_=True)
            self.image_w_grad_original = repreprocessed_img
            pixmap_grad = QPixmap(repreprocessed_img)
            if pixmap_grad.isNull():
                QMessageBox.critical(self, "Image Error", "Transition to grad failed.")
                return None
            scale_pixmap_grad = pixmap_grad.scaledToHeight(480, Qt.SmoothTransformation)
            self.image.setPixmap(scale_pixmap_grad)
        else:
            pixmap = QPixmap(self.img_path)
            if pixmap.isNull():
                QMessageBox.critical(self, "Image Error", "Image not loaded properly. Check extension")
                return None
            scale_pixmap = pixmap.scaledToHeight(480, Qt.SmoothTransformation)
            self.image.setPixmap(scale_pixmap)