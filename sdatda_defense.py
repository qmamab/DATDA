# ==============================================================================
# SDATDA — Secondary Defense Against The Dark Arts (Ultra Defense)
# Author: Qamar Muneer Akbar
# ORCID: 0009-0003-6671-9253 | www.ftiuae.com
# Academic Use Only — Non-commercial research with attribution.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

from datda_index import DATDAIndex, _to_tensor, _from_pil  # re-use index utilities
from datda_defense import _dct_2d, _idct_2d, tv_denoise_torch  # reuse defense helpers

device = "cuda" if torch.cuda.is_available() else "cpu"

class SDATDAConfig:
    """Config for SDATDA ultra defense."""
    def __init__(
        self,
        spectral_aggressiveness: float = 0.85,
        tv_iters: int = 16,
        tv_weight: float = 0.25,
        gradient_shield_sigma: float = 0.025,
        iterative_steps: int = 5,
        iterative_eps: float = 0.05,
        iterative_step_size: float = 0.01,
        momentum_decay: float = 0.9,
        seed: int = 1234
    ):
        self.spectral_aggressiveness = spectral_aggressiveness
        self.tv_iters = tv_iters
        self.tv_weight = tv_weight
        self.gradient_shield_sigma = gradient_shield_sigma
        self.iterative_steps = iterative_steps
        self.iterative_eps = iterative_eps
        self.iterative_step_size = iterative_step_size
        self.momentum_decay = momentum_decay
        self.seed = seed

torch.manual_seed(1234)
np.random.seed(1234)

# ------------------------------
# SDATDA Defense Class
# ------------------------------
class SDATDAUltra:
    """
    Secondary Defense Pipeline (SDATDA)
    Extremely aggressive purification for adversarial images.
    """
    def __init__(self, config: SDATDAConfig = SDATDAConfig()):
        self.config = config
        self.index_calc = DATDAIndex()

    # --------------------
    # Spectral Extreme Suppression
    # --------------------
    def spectral_extreme(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        for b in range(B):
            for c in range(C):
                ch = x[b:b+1, c:c+1]
                fft = torch.fft.fft2(ch.squeeze(0))
                fft_shift = torch.fft.fftshift(fft)
                mag = torch.abs(fft_shift)
                phase = torch.angle(fft_shift)
                center_h, center_w = H // 2, W // 2
                Y, X = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
                dist = torch.sqrt((X - center_w).float()**2 + (Y - center_h).float()**2)
                radius = int(min(H, W) * 0.25)
                high_mask = (dist > radius).float()
                mag = mag * (1.0 - self.config.spectral_aggressiveness * high_mask)
                fft_clean = mag * torch.exp(1j * phase)
                ifft_shift = torch.fft.ifftshift(fft_clean)
                recon = torch.fft.ifft2(ifft_shift).real
                out[b:b+1, c:c+1] = recon.unsqueeze(0)
        return out.clamp(0.0, 1.0)

    # --------------------
    # Total Variation + JPEG + Gradient Shield
    # --------------------
    def tv_jpeg_shield(self, x: torch.Tensor) -> torch.Tensor:
        # TV Denoise
        out = torch.cat([tv_denoise_torch(x[i:i+1], weight=self.config.tv_weight, iters=self.config.tv_iters) for i in range(x.shape[0])], dim=0)
        # Gentle Gaussian Blur
        out = gaussian_blur(out, kernel_size=3, sigma=0.8)
        # Gradient Shield (stochastic noise)
        noise = torch.randn_like(out) * self.config.gradient_shield_sigma
        out = (out + noise).clamp(0.0, 1.0)
        return out

    # --------------------
    # Iterative Momentum-based Purification
    # --------------------
    def iterative_purify(self, x: torch.Tensor, surrogate_model=None) -> torch.Tensor:
        """
        Optional iterative reverse attack for super aggressive defense.
        surrogate_model: dummy or real CNN model to estimate gradients
        """
        x_iter = x.clone().detach()
        momentum = torch.zeros_like(x_iter)
        eps = self.config.iterative_eps
        step = self.config.iterative_step_size
        decay = self.config.momentum_decay

        for _ in range(self.config.iterative_steps):
            x_iter.requires_grad = True
            if surrogate_model:
                logits = surrogate_model(x_iter)
                preds = logits.argmax(dim=1)
                loss = F.cross_entropy(logits, preds)
                grad = torch.autograd.grad(loss, x_iter)[0]
            else:
                # No surrogate: use simple high-frequency gradient
                grad = torch.fft.ifft2(torch.fft.fft2(x_iter) * 1j).real
            momentum = decay * momentum + grad / (grad.abs().mean() + 1e-12)
            x_iter = (x_iter - step * torch.sign(momentum)).clamp(0.0, 1.0).detach()
        return x_iter

    # --------------------
    # Single-pass SDATDA
    # --------------------
    def purify(self, image: Image.Image) -> (Image.Image, float):
        """
        image: PIL.Image
        returns: defended PIL.Image, final DATDA Index
        """
        x = _to_tensor(image)
        # Stage 1: Spectral extreme
        x = self.spectral_extreme(x)
        # Stage 2: TV + JPEG + Shield
        x = self.tv_jpeg_shield(x)
        # Stage 3: Iterative purification (momentum)
        x = self.iterative_purify(x)
        # Compute final DATDA Index
        final_index = self.index_calc.compute(x)
        # Convert back to PIL
        defended = _from_pil(Image.fromarray((x[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)))
        return defended, final_index


# --------------------
# Quick test/demo
# --------------------
if __name__ == "__main__":
    img_path = "test.jpg"  # replace with your test image
    img = Image.open(img_path).convert("RGB")
    sdatda = SDATDAUltra()
    defended_img, idx = sdatda.purify(img)
    print(f"Final DATDA Index after SDATDA: {idx:.4f}")
    defended_img.show()
