import numpy as np
import cv2
from typing import Tuple

# import modules heere:
########## MODULES ###########
from keras.applications.vgg16 import preprocess_input
########## ENDMODULES ###########

# decorator for checking input and output dimensions, type
def handle_io_type(func):
    def wrapper(image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image to match the required model input format.

        Args:
            image (np.ndarray): Input image as a uint8 array with shape (x, y, 3).

        Returns:
            np.ndarray: Preprocessed image as a float32 array with shape (1, 224, 224, 3).
            Batch dimension will be added automatically (axis 0)
        """
        # Check input type and shape
        assert image.dtype == np.uint8, "Input image must be of type uint8"
        assert len(image.shape) == 3 and image.shape[2] == 3, "Input image must have shape (x, y, 3)"
        
        preprocessed_image = func(image)
        
        return np.expand_dims(preprocessed_image, axis=0)
    return wrapper


@handle_io_type
# place code of your preprocessing function here:
########## CODE ###########

def custom_preprocessing(image):
    image_resized = cv2.resize(image, (224, 224))
    image_as_array = np.array(image_resized)
    image_preprocessed = preprocess_input(image_as_array)
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension
    return image_preprocessed

########## ENDCODE ########### 


