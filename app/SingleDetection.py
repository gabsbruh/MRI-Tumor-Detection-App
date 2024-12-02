from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, 
    QLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from app.BaseDetection import BaseDetection
# import functions one by one to prevent import whole library (reduce size of .exe)
from cv2 import imread as cv2_imread
from numpy import argmax as np_argmax
### modules lazy-loaded
# from script.GradCAM import GradCAM


class SingleDetection(BaseDetection):
    def __init__(self, show_page_callback):
        super().__init__(show_page_callback)
        
        # no-ui instances
        self.img_path = ""
        self.original_image = None
        self.image_w_grad_original = None
        self.detection_result = None
        
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
            # reset buttons and variables to default
            self.grad_cam.setChecked(False)
            self.result.setText("Detection Results")
            self.detection_result = None
            # set image and resize properly to display
            self.original_image = cv2_imread(self.img_path) # cv2.imread to load original image
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
        if self.model is None:
            QMessageBox.information(self, "Model Loading", "Model are being loaded. When model info will appear, try again.")
            return None
        if self.original_image is None:
            QMessageBox.warning(self, "Load Image", "Load image to use detect.")
            return None
        self.result.setText("")
        self.result.update()
        preprocessed_img = self.preprocess_image(self.original_image)
        output = self.model.predict(preprocessed_img)
        if np_argmax(output) == 1:
            self.detection_result = 1
            self.result.setText("DETECTED")
            self.result.setObjectName("TUMOR_DETECTED")
            self.image.setObjectName("TUMOR_DETECTED_IMAGE")
        else:
            self.detection_result = 0
            self.result.setText("No Tumor")
            self.result.setObjectName("NO_TUMOR")
            self.image.setObjectName("NO_TUMOR_IMAGE")
        # reload styling to apply to result
        self.style().polish(self.result)
        self.style().polish(self.image)
    
    def apply_grad_cam(self, checked):
        if self.original_image is None:
            QMessageBox.warning(self, "Load Image", "Load an image before applying Grad-CAM.")
            return
        
        if self.detection_result is None:
            self.grad_cam.setChecked(False)    
            QMessageBox.warning(self, "Detect Tumor", "Use 'Detect' button to detect tumor first.")
            return None
        
        if self.grad_cam_alg is None:
            # lazy-loading 
            from script.GradCAM import GradCAM
            # after loading model, grad-cam alg can be initialized by composition
            self.grad_cam_alg = GradCAM(self.model)   

        if checked:
            # use grad cam class for getting image with cnn's activation areas applied
            overlay_image = self.grad_cam_alg.create_gradcam_image(self.img_path, class_index=self.detection_result)
            self.image_w_grad_original = overlay_image
            overlayed_qimage = self.preprocess_image(self.image_w_grad_original, reversed_=True)
            pixmap_grad = QPixmap(overlayed_qimage)
            if pixmap_grad.isNull():
                QMessageBox.critical(self, "Image Error", "Transition to Grad-CAM failed.")
                return
            scale_pixmap_grad = pixmap_grad.scaledToHeight(480, Qt.SmoothTransformation)
            self.image.setPixmap(scale_pixmap_grad)
        else:
            pixmap = QPixmap(self.img_path)
            if pixmap.isNull():
                QMessageBox.critical(self, "Image Error", "Image not loaded properly. Check the extension.")
                return
            scale_pixmap = pixmap.scaledToHeight(480, Qt.SmoothTransformation)
            self.image.setPixmap(scale_pixmap)
