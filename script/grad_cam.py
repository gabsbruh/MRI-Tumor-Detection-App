import numpy as np
from tensorflow.keras.layers import Conv2D, SeparableConv2D
from tensorflow.keras.models import Model
from tensorflow import GradientTape
from tensorflow import reduce_mean, reduce_sum, float32, convert_to_tensor
from tensorflow.image import resize
from matplotlib.pyplot import get_cmap

class GradCam:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation to generate heatmaps 
    highlighting important regions in an image for a specific class using a convolutional neural network.
    """

    def __init__(self, model):
        self.model = model
        self.last_conv_layer_name = None
        self.find_last_conv_layer_name()
    
    def find_last_conv_layer_name(self):
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name or isinstance(Conv2D, 
                                                  SeparableConv2D):
               self.last_conv_layer_name = layer.name
               break
        if not self.last_conv_layer_name:
            print("conv layer not found and set to default: 'block5_conv3'")
            self.last_conv_layer_name = 'block5_conv3'
                
    
    def compute_heatmap(self, image, class_idx):
        """
        Compute the Grad-CAM heatmap for a given image and class index.
        
        Args:
            image (np.ndarray): Input image formatted to match the model's input shape.
            class_idx (int): Index of the class for which Grad-CAM heatmap is computed.

        Returns:
            np.ndarray: Normalized Grad-CAM heatmap as a 2D NumPy array.
        """
        # Model consisting of the convolutional layer and output
        grad_model = Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        
        # Compute gradients with respect to the class index
        with GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)  # Gradients with respect to conv layer output
        pooled_grads = reduce_mean(grads, axis=(0, 1, 2))  # Average gradients
        
        # Weight activation maps with gradients
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        heatmap = reduce_sum(pooled_grads * conv_outputs, axis=-1)
        
        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.8, cmap="jet"):
        """
        Overlay the Grad-CAM heatmap on the original image.

        Args:
            heatmap (np.ndarray): Grad-CAM heatmap as a 2D array.
            image (np.ndarray): Original image in RGB format, scaled to [0, 255].
            alpha (float, optional): Transparency level of the heatmap overlay. Defaults to 0.4.
            cmap (str, optional): Colormap to use for the heatmap. Defaults to "viridis".

        Returns:
            np.ndarray: Image with the heatmap overlay applied.
        """
        # Scale heatmap to 0-255
        heatmap = np.uint8(255 * heatmap)
        colormap = get_cmap(cmap)
        heatmap_colored = colormap(heatmap)
        heatmap_colored = np.delete(heatmap_colored, 3, axis=-1)  # Remove alpha channel
        
        # Resize heatmap to match image dimensions
        heatmap_colored = resize(heatmap_colored, (image.shape[1], image.shape[2]))
        heatmap_colored = convert_to_tensor(heatmap_colored.numpy(), dtype=float32)
        
        # Blend heatmap and original image
        overlay = heatmap_colored * alpha + image / 255.0
        return np.clip(overlay, 0, 1)
