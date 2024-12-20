from cv2 import (
    imread as cv2_imread,
    cvtColor as cv2_cvtColor,
    resize as cv2_resize,
    COLOR_BGR2RGB as cv2_COLOR_BGR2RGB,
    COLORMAP_JET as cv2_COLORMAP_JET,
    applyColorMap as cv2_applyColorMap,
    addWeighted as cv2_addWeighted
)
from numpy import (
    expand_dims as np_expand_dims,
    uint8 as np_uint8,
    dot as np_dot,
    maximum as np_maximum,
    max as np_max
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Conv2D, SeparableConv2D
from keras.activations import softmax as keras_activations_softmax
from tensorflow import GradientTape as tf_GradientTape
from tensorflow import reduce_mean as tf_reduce_mean

class GradCAM():
    def __init__(self, model):
        """
        Computes the Grad-CAM heatmap and overlays it on the original image.

        Args:
            model: Keras model to classify image
        """
        self.model = model
        self.last_conv_layer_name = self._find_last_conv_layer_name()
        self._disable_softmax()

    def _find_last_conv_layer_name(self):
        """
        Finds the name of the last convolutional layer in the model.

        Returns:
            str: The name of the last convolutional layer.
        """
        # start from the end 
        for layer in reversed(self.model.layers):
            # validate type of layer
            if 'conv' in layer.name or isinstance(layer, (Conv2D, SeparableConv2D)):
                return layer.name
        # case for not founding
        print("Conv layer not found; setting to default: 'block5_conv3'")
        return 'block5_conv3'

    def _load_model(self):
        """Load the model from a file."""
        return load_model(self.model_path)

    def _disable_softmax(self):
        """Disable the softmax activation in the last layer."""
        if hasattr(self.model.layers[-1], "activation") and self.model.layers[-1].activation == keras_activations_softmax:
            self.model.layers[-1].activation = None

    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Load and preprocess an image.

        Args:
            image_path (str): Path to the image.
            target_size (tuple): Size to resize the image to.

        Returns:
            np.ndarray: Preprocessed image ready for the model.
        """
        img = cv2_imread(image_path)
        img = cv2_cvtColor(img, cv2_COLOR_BGR2RGB)
        img = cv2_resize(img, target_size)
        img = preprocess_input(img)  # Using preprocess_input from EfficientNet
        return np_expand_dims(img, axis=0)

    def generate_heatmap(self, image, class_index):
        """
        Generate a Grad-CAM heatmap for a selected image and class.

        Args:
            image (np.ndarray): Image preprocessed for the model.
            class_index (int): Class index for which the heatmap is generated.

        Returns:
            np.ndarray: Normalized heatmap.
        """
        grad_model = Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )
        
        with tf_GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf_reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = np_dot(conv_outputs, pooled_grads.numpy())
        heatmap = np_maximum(heatmap, 0)  # ReLU
        if np_max(heatmap) > 0:
            heatmap /= np_max(heatmap)  # Normalization

        return heatmap

    def overlay_heatmap(self, heatmap, original_image_path, alpha=0.4, colormap=cv2_COLORMAP_JET):
        """
        Overlay the heatmap on the original image.

        Args:
            heatmap (np.ndarray): Grad-CAM heatmap.
            original_image_path (str): Path to the original image.
            alpha (float): Transparency coefficient for the heatmap.
            colormap (int): Colormap for the heatmap.

        Returns:
            np.ndarray: Image with the heatmap overlay.
        """
        original_image = cv2_imread(original_image_path)
        original_image = cv2_cvtColor(original_image, cv2_COLOR_BGR2RGB)

        heatmap_resized = cv2_resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_resized = 1 - heatmap_resized
        heatmap_resized = np_uint8(255 * heatmap_resized)
        heatmap_colored = cv2_applyColorMap(heatmap_resized, colormap)

        overlayed_image = cv2_addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlayed_image

    def create_gradcam_image(self, image_path, class_index=1):
        """
        Full process: image preprocessing, heatmap generation, heatmap overlay, and display.

        Args:
            image_path (str): Path to the input image.
            class_index (int): Class index for calculations.

        """
        # Preprocess image and generate heatmap
        image = self.preprocess_image(image_path)
        heatmap = self.generate_heatmap(image, class_index)
        overlayed_image = self.overlay_heatmap(heatmap, image_path)
        return overlayed_image

