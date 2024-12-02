from script.GradCAM import GradCAM
from keras.models import load_model
import numpy as np


def test_find_last_conv_layer():
    model = load_model("models/EfficientNet.keras")
    grad_cam = GradCAM(model)
    
    assert grad_cam.last_conv_layer_name == "top_conv"

def test_generate_heatmap():
    model = load_model("models/EfficientNet.keras")
    grad_cam = GradCAM(model)
    dummy_image = np.ones((1, 224, 224, 3), dtype=np.float32)
    heatmap = grad_cam.generate_heatmap(dummy_image, class_index=1)
    assert heatmap.shape != ()