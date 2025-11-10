# ==============================================================================
# DATDA-INDEX: Adversarial Perturbation Severity Estimator
# Author: Qamar Muneer Akbar
# ORCID: 0009-0003-6671-9253 | www.ftiuae.com
# Description:
#     Computes the DATDA Index, a quantitative measure (0-1)
#     representing adversarial perturbation or anomaly strength in images.
#     Designed to run independently before/after DATDA defense.
# ==============================================================================

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
import math


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def _to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor on CPU/GPU."""
    return ToTensor()(img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# DATDA Index Calculator
# ------------------------------------------------------------

class DATDAIndex:
    """
    Computes a DATDA Index score between 0 and 1 indicating perturbation severity.
    Combines spectral energy, local variance, and pixel distribution metrics.
    """

    def __init__(self, high_freq_ratio: float = 0.35):
        self.high_freq_ratio = high_freq_ratio

    @torch.no_grad()
    def compute(self, x: torch.Tensor) -> float:
        """
        Args:
            x (torch.Tensor): [1x3xHxW] image tensor, values in [0,1]
        Returns:
            float: DATDA Index (0.0 to 1.0)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)

        B, C, H, W = x.shape
        device = x.device

        # Convert to grayscale
        x_gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)

        # 1️⃣ High-frequency spectral ratio
        fft = torch.fft.fft2(x_gray.squeeze(1))
        fft_shift = torch.fft.fftshift(fft)
        mag = torch.abs(fft_shift)
        center_h, center_w = H // 2, W // 2
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        dist = torch.sqrt((xx - center_w) ** 2 + (yy - center_h) ** 2)
        radius = int(self.high_freq_ratio * min(H, W))
        high_mask = (dist > radius).float()
        high_energy = (mag * high_mask).sum()
        total_energy = mag.sum() + 1e-8
        freq_ratio = high_energy / total_energy

        # 2️⃣ Local variance
        pad = 1
        x_pad = F.pad(x_gray, (pad, pad, pad, pad), mode="reflect")
        patches = x_pad.unfold(2, 3, 1).unfold(3, 3, 1)
        var_local = (patches - patches.mean(dim=(-1, -2), keepdim=True)).pow(2).mean()
        var_score = torch.clamp(var_local * 2.0, 0, 1)

        # 3️⃣ Residual noise vs smoothed baseline
        baseline = F.avg_pool2d(x_gray, kernel_size=3, stride=1, padding=1)
        res = (x_gray - baseline).abs().mean()

        # 4️⃣ Entropy
        flat = (x_gray.view(-1) * 255).long().clamp(0, 255)
        hist = torch.bincount(flat, minlength=256).float()
        probs = hist / (hist.sum() + 1e-8)
        entropy = -(probs * (probs + 1e-12).log()).sum() / math.log(256.0)

        # Combine features (weighted)
        weights = torch.tensor([0.35, 0.25, 0.25, 0.15], device=device)
        feats = torch.tensor([freq_ratio, var_score, res, entropy], device=device)
        feats_norm = (feats - feats.min()) / (feats.max() - feats.min() + 1e-8)
        index = (feats_norm * weights).sum().item()

        return round(float(np.clip(index, 0, 1)), 4)


# ------------------------------------------------------------
# Example CLI use
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python datda-index.py <image_path>")
        exit(0)

    img_path = sys.argv[1]
    img = Image.open(img_path).convert("RGB")

    calc = DATDAIndex()
    x = _to_tensor(img)
    idx = calc.compute(x)
    print(f"DATDA Index: {idx:.4f}")
