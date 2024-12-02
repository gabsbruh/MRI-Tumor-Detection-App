from typing import Tuple
from numpy import (
    uint8 as np_uint8,
    expand_dims as np_expand_dims,
    ndarray as np_ndarray,
    array as np_array
)
from cv2 import resize as cv2_resize

# import modules heere:
########## MODULES ###########
from keras.applications.vgg16 import preprocess_input
########## ENDMODULES ###########

# decorator for checking input and output dimensions, type
def handle_io_type(func):
    def wrapper(image: np_ndarray) -> np_ndarray:
        """
        Preprocess the input image to match the required model input format.

        Args:
            image (np.ndarray): Input image as a uint8 array with shape (x, y, 3).

        Returns:
            np.ndarray: Preprocessed image as a float32 array with shape (1, 224, 224, 3).
            Batch dimension will be added automatically (axis 0)
        """
        # Check input type and shape
        assert image.dtype == np_uint8, "Input image must be of type uint8"
        assert len(image.shape) == 3 and image.shape[2] == 3, "Input image must have shape (x, y, 3)"
        
        preprocessed_image = func(image)
        
        return np_expand_dims(preprocessed_image, axis=0)
    return wrapper


@handle_io_type
# place code of your preprocessing function here:
########## CODE ###########

def custom_preprocessing(image):
    image_resized = cv2_resize(image, (224, 224))
    image_as_array = np_array(image_resized)
    image_preprocessed = preprocess_input(image_as_array)
    return image_preprocessed

########## ENDCODE ########### 


