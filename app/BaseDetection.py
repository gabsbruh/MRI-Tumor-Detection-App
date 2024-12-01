import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, 
    QVBoxLayout, QFileDialog, QLineEdit,
    QMessageBox
)
from PyQt5.QtGui import QIcon, QImage
from threading import Thread
### modules lazy-loaded
# from keras.models import load_model
# from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
# from script.preprocess_input_custom import custom_preprocessing


class BaseDetection(QWidget):
    """instances:
            go_back: return button to page before
            browser: button for uploading photos
            upload_model: button for uploading custom model
            model_info: label about model in use
            path: blocked lineedit to show current paht of the photo
        
        no-ui instances:
            model: instance holds model which classifies images
            original_image: initialization. holds image displayed
            model_is_default: bool, flag for preprocessing
        
        methods (To Implement!):
            setup_interface: Organize layout of widgets
            detect_tumor: Write a function which runs cnn for classification
            browse_for_img: Organize retrieving image from file
            apply_grad_cam: Apply grad cam to the image
        
        methods already implemented:
            load_model_: loads weights of neural network to the application
            save_image: saves image with grad-cam
            preprocess_image: Preprocess image for model predicting
    """
    # singals are defined or class level
    update_model_info = pyqtSignal(str)
    
    def __init__(self, show_page_callback):
        super().__init__()
        
        # callback to return to start page
        self.show_page = show_page_callback
        
        # info about model - double-rowed qlabel
        self.model_info = QLabel("model is loading...", alignment=Qt.AlignCenter)
        self.model_info.setFixedHeight(40)
        
        # icon
        go_back_icon = QIcon("static/img/return.png")
        self.go_back = QPushButton()
        self.go_back.setIcon(go_back_icon)
        self.go_back.setIconSize(QSize(40, 20))
        self.go_back.setFixedSize(35,35)

        # rest base widgets
        # browser
        self.browser = QPushButton("browse")
        self.browser.setFixedSize(180, 35)
        
        # grad cam alg
        self.grad_cam = QPushButton("grad-cam")
        self.grad_cam.setCheckable(True)
        self.grad_cam.setObjectName('GRAD_CAM_BTN')
        self.grad_cam.setFixedSize(150, 50)
        
        # saving grad image
        self.save_grad = QPushButton("save image")
        self.save_grad.setFixedSize(150, 50)

        # upload custom model
        self.upload_model = QPushButton("upload own model")
        self.upload_model.setFixedSize(200, 40)
        
        # path to image show
        self.path = QLineEdit(self)
        self.path.setReadOnly(True)
        self.path.setFixedSize(300, 40)
        # keep path at the end if too long
        self.path.setAlignment(Qt.AlignRight)
        
        # detect tumor button
        self.detect = QPushButton("DETECT")
        self.detect.setFixedSize(300,50)
        self.detect.setObjectName('DETECT_BTN')
        
        # initialize image object to use in this class
        self.original_image = None 
        self.image_w_grad_original = None
        
        # grad cam alg
        self.grad_cam_alg = None
        
        # load CNN model
        self.model_is_default = None
        self.model = None
        
        # connect buttons
        self.browser.clicked.connect(self.browse_for_img)
        self.go_back.clicked.connect(lambda: self.show_page("Start"))
        self.upload_model.clicked.connect(lambda: self.load_model_(default_model=False))
        self.grad_cam.toggled.connect(self.apply_grad_cam)
        self.save_grad.clicked.connect(self.save_image)
        self.detect.clicked.connect(self.detect_tumor)
        self.update_model_info.connect(self.update_model_info_label)
        
    def lazy_loading(self):
        """Few modules are time-consuming to load. load them when they are needed.
        """
        def start_loading():
            # lazy-loading process of loading model
            from keras.models import load_model
            self.load_model_()  
        Thread(target=start_loading, daemon=True).start()        
    
    def load_model_(self, default_model=True):
        """
            Load model of cnn to the application as '.h5' or '.keras' file. Additionaly, change model info - name, by taking name of the file.
        Args:
            default_model (bool, optional): Remain True if default model should be loaded. Instead, Opens '.h5' or '.keras' model to be loaded.

        Returns:
            NoneType: return None if an error occur.
        """
        # lazy-loading
        from keras.models import load_model
        model_info_func = lambda mt, mn: ("<html><body>"
                                          f"<p style='font-size:12px;margin:0;'>model: {mt},</p>"
                                          f"<p style='margin:0;'>{mn}</p>"
                                          "</body></html>")
        if  default_model:
            self.model = load_model("models/EfficientNet.keras", compile=False)
            model_type = 'default'
            model_name = 'EfficientNet'
            self.model_is_default = True
        
        else:
            options = QFileDialog.Options()
            filepath, _ = QFileDialog.getOpenFileName(self, 
                                                "Select file", "", 
                                                "Model Files (*.h5 *.keras);;All Files (*)", 
                                                options=options)
            if filepath:
                try:
                    self.model_is_default = False
                    self.model = load_model(filepath)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not load model: {e}")
                    return None
                else:
                    model_type = 'custom'
                    model_name = filepath.split('/')[-1].split('.')[0]
            else:
                QMessageBox.warning(self, "No file chosen", "Choose '.h5' or '.keras.' file.")
                return None
        self.update_model_info.emit(model_info_func(model_type, model_name))
    
    def update_model_info_label(self, text):
        """Update text in model_info label. Func is required due to model loading"""
        self.model_info.setText(text)
    
    def save_image(self):
        """save image which is currently displayed
        """
        filepath, _ = QFileDialog.getSaveFileName(self, 
                                                "Choose Directory and filename", 
                                                "", 
                                                "Image Files (*.png *.jpg *.jpeg *.svg *.bmp)"
                                                )
        if filepath:
            try: 
                image_bgr = cv2.cvtColor(self.image_w_grad_original, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, image_bgr)
                QMessageBox.information(self, "Success", f"Image saved in {filepath}.")
            except Exception as e:
                try: 
                    image_pixmap = self.original_image
                    if image_pixmap.save(filepath):
                        QMessageBox.information(self, "Success", f"Image saved in {filepath}.")
                    else:
                        QMessageBox.warning(self, "Error", f"An error occured: {e}")
                except AttributeError as e:
                    # if method pixmap returns nonetype
                    QMessageBox.warning(self, "Error", f"An error occured: {e}")
                    return None
        else:
            QMessageBox.warning(self, "No directory chosen", "Choose directory")
            return None            
            
    def preprocess_image(self, image, reversed_=False):
        """
        Resize and prepare image for prediction or reverse the process for display.

        Args:
            image (ndarray or QImage): The image to preprocess.
            reversed_ (bool): If True, convert the image back for display.

        Returns:
            ndarray or QImage: Preprocessed image or reversed QImage.
        """
        if reversed_:
            if len(image.shape) == 3:  # Handle (height, width, channels) format
                h, w, c = image.shape
                image = np.clip(image, 0, 255).astype(np.uint8)  # Ensure valid pixel range
                bytes_per_line = 3 * w
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                return q_image
            elif len(image.shape) == 4:  # Handle (batch, height, width, channels)
                _, h, w, c = image.shape
                image = (image[0] * 255).astype(np.uint8)  # Remove batch dimension
                bytes_per_line = 3 * w
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                return q_image
            else:
                raise ValueError("Unsupported image shape for reversed operation.")
        else:
            if self.model_is_default:
                # lazy-loading of preprocessing
                from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
                image_resized = cv2.resize(image, (224, 224))
                image_as_array = np.array(image_resized)
                image_preprocessed = efficientnet_preprocess_input(image_as_array)
                image_preprocessed = np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension
                return image_preprocessed
            else:
                try:
                    # lazy-loading of custom-preprocessing
                    from script.preprocess_input_custom import custom_preprocessing
                    image_preprocessed = custom_preprocessing(image)
                except Exception as e:
                    QMessageBox.critical(self, "Preprocess func error", f"Custom preprocessing failed: {e}")
                return image_preprocessed
                
    def apply_grad_cam(self, checked):
        """Prebuilt method for grad_cam optional
        """
        pass
    
    def setup_interface(self):
        raise NotImplementedError("Organize layout of widgets")
        
    def detect_tumor(self):
        raise NotImplementedError("Write a function which runs cnn for classification")
    
    def browse_for_img(self):
        raise NotImplementedError("Organize retrieving image from file")
