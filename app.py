# ==============================================================================
# Main Gradio App for DATDA + SDATDA Defense with CNN Classification Ensemble
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import gradio as gr
import numpy as np

from datda_defense import DATDA, DATDAConfig  # primary defense
from datda_index import DATDAIndex  # index calculator
from sdatda_defense import SDATDAUltra, SDATDAConfig  # secondary ultra-defense

# Import torchvision models
import torchvision.models as models
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Image preprocessing transforms
# ------------------------------
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# ------------------------------
# Load candidate CNN models
# ------------------------------
MODEL_CATALOG = {
    "resnet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval().to(device),
    "vgg16": models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval().to(device),
    "efficientnet_b0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).eval().to(device),
    "mobilenet_v3_large": models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).eval().to(device)
}

# Model robustness scores (higher = more robust to adversarial perturbations)
MODEL_ROBUSTNESS = {
    "resnet50": 0.8,
    "vgg16": 0.6,
    "efficientnet_b0": 0.75,
    "mobilenet_v3_large": 0.7
}

# ------------------------------
# Initialize defenses and index
# ------------------------------
datda_model = DATDA(DATDAConfig(device=device))
sdatda_model = SDATDAUltra(SDATDAConfig())
index_calc = DATDAIndex()

# ------------------------------
# Ensemble model selection function
# ------------------------------
def ensemble_cnn_predict(datda_index: float, img: torch.Tensor, top_k=2):
    """
    Ensemble CNN predictions using robustness and DATDA index as weights.
    """
    # Compute HF ratio of image
    x_gray = (0.299 * img[0,0] + 0.587 * img[0,1] + 0.114 * img[0,2])
    fft = torch.fft.fft2(x_gray)
    fft_shift = torch.fft.fftshift(fft)
    mag = torch.abs(fft_shift)
    H, W = x_gray.shape
    center_h, center_w = H//2, W//2
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = torch.sqrt((X - center_w).float()**2 + (Y - center_h).float()**2)
    radius = int(min(H, W) * 0.25)
    hf_ratio = (mag[dist > radius].sum() / (mag.sum() + 1e-8)).item()

    # Compute weights for each model
    model_weights = {}
    for name, robustness in MODEL_ROBUSTNESS.items():
        weight = robustness * (1 - datda_index + hf_ratio)  # higher weight = more confident
        model_weights[name] = weight

    # Pick top_k models for ensemble
    selected_models = sorted(model_weights, key=model_weights.get, reverse=True)[:top_k]

    # Compute weighted ensemble predictions
    logits_sum = None
    total_weight = 0.0
    for name in selected_models:
        model = MODEL_CATALOG[name]
        weight = model_weights[name]
        with torch.no_grad():
            logits = model(img)
        if logits_sum is None:
            logits_sum = weight * F.softmax(logits, dim=1)
        else:
            logits_sum += weight * F.softmax(logits, dim=1)
        total_weight += weight

    ensemble_probs = logits_sum / total_weight
    top_prob, top_class = ensemble_probs.max(dim=1)

    return top_class.item(), top_prob.item(), selected_models

# ------------------------------
# Main processing pipeline
# ------------------------------
def process_image(img: Image.Image):
    # Preprocess to tensor
    x_tensor = preprocess(img).unsqueeze(0).to(device)

    # Step 1: Initial DATDA Index
    index_before = index_calc.compute(x_tensor)

    # Step 2: Run DATDA defense
    x_defended = datda_model(x_tensor)
    index_after = index_calc.compute(x_defended)

    # Step 3: If still high, run SDATDAUltra
    if index_after > 0.3:  # threshold can be tuned
        x_defended_img, index_final = sdatda_model.purify(img)
        x_defended = preprocess(x_defended_img).unsqueeze(0).to(device)
    else:
        index_final = index_after

    # Step 4: Ensemble CNN classification
    top_class, top_prob, selected_models = ensemble_cnn_predict(index_final, x_defended)

    # Convert back to PIL for display
    img_defended = T.ToPILImage()(x_defended[0].cpu())

    return img_defended, float(index_before), float(index_final), ", ".join(selected_models), float(top_prob)

# ------------------------------
# Gradio UI
# ------------------------------
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Defended Image"),
        gr.Number(label="Initial DATDA Index"),
        gr.Number(label="Final DATDA Index"),
        gr.Textbox(label="Selected CNN Models (Ensemble)"),
        gr.Number(label="Classification Confidence")
    ],
    title="DATDA + SDATDA Defense with CNN Ensemble",
    description="Upload an image. The system applies the primary DATDA defense, optionally the SDATDAUltra defense, then uses an ensemble of CNNs selected adaptively to classify the image."
)

if __name__ == "__main__":
    iface.launch()
