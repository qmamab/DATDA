# DATDA — Hybrid Defense & Image Classification Model

## Overview

DATDA (Dynamic Adversarial Training & Defense Architecture) is a cutting-edge hybrid model designed to simultaneously defend and classify images. It moves beyond standard single-model solutions by combining adaptive adversarial defenses with a dynamic model routing mechanism, ensuring high classification accuracy even when facing sophisticated adversarial attacks.

## Key Features

- **Hybrid Defense & Classification**: Functions as both a robust protection mechanism and a high-accuracy classification engine within a unified framework.

- **Smart Model Routing with DATDA Index**: Computes a unique DATDA Index (pre- and post-defense) to intelligently gauge attack severity. This index dynamically selects the optimal CNN model for the image:
  - Robust Legacy Models (e.g., VGG16) for highly corrupted images.
  - Advanced Hybrid CNNs for clean or partially defended images.

- **Adaptive Purification Paths**: A multi-strategy defense that actively removes adversarial noise using a variety of complementary techniques:
  - Spectral Suppression & DCT Low-Pass Filtering
  - Bilateral/Median Smoothing & Total Variation Denoising
  - Gradient Shielding & Random Augmentation

- **Secondary Ultra-Defense (SDATDA)**: An optional, intensified defense layer that automatically activates for images with a high post-defense index. It combines momentum, gradient, and reverse techniques to neutralize deep-seated perturbations.

- **Gradio & Hugging Face Ready**: Designed for seamless integration into interactive web demos or research pipelines for immediate testing and deployment.

## Installation

Clone the repository and install as a Python package.


Install using PIP 


```bash
pip install datda
```

## Quick Usage

Defend and classify an image in four simple steps.

```python
from DATDA import DATDA, SDATDA, DATDAIndex
from PIL import Image

# 1. Load Image and Compute Pre-defense Index
img = Image.open("example.png")
index_calc = DATDAIndex()
pre_index = index_calc.compute_index(img)
print(f"Pre-defense DATDA Index: {pre_index}")

# 2. Apply Primary Hybrid Defense & Classification
datda = DATDA()
defended_img, predicted_label = datda(img)

# 3. Compute Post-defense Index
post_index = index_calc.compute_index(defended_img)
print(f"Post-defense DATDA Index: {post_index}")

# 4. Optional: Invoke Secondary Ultra-Defense (SDATDA)
# Use a threshold to decide if extra defense is needed
if post_index > 0.5:
    print("Activation SDATDA...")
    sdatda = SDATDA()
    cleaned_img = sdatda(defended_img)
    _, predicted_label = datda(cleaned_img) # Re-classify the ultra-cleaned image

print(f"Final Predicted Label: {predicted_label}")
```

## Project Structure

```
DATDA/
├── __init__.py
├── datda_defense.py      # Core Primary Hybrid Defense + Classification Logic
├── sdatda_defense.py     # Secondary Ultra-Defense Implementation
├── datda_index.py        # DATDA Index Calculator for Smart Routing
├── app.py                # Example Gradio Web UI
├── setup.py
└── README.md
```

## Citation & Academic Use

This library is intended for academic and research purposes only. If you use DATDA in a publication, please cite the author:

```bibtex
@misc{akbar2025datda,
  author = {Qamar Muneer Akbar},
  title = {DATDA: Hybrid Defense & Image Classification},
  year = {2025},
  url = {https://www.ftiuae.com}
}
```

## Fun Fact

The name **DATDA** was inspired by the *Defense Against the Dark Arts* at Hogwarts. 

---

Made by **Qamar Muneer Akbar** and pinch of magic by Gemini2.5-Pro
