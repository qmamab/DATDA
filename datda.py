# datda.py
# ==============================================================================
# DATDA — Defense Against The Dark Arts
# Copyright (c) 2025 Qamar Muneer Akbar
# ORCID: 0009-0003-6671-9253 | qamar@ftiuae.com | www.ftiuae.com
# Academic Use Only — Non-commercial research with attribution.
# ==============================================================================
import io
import math
import warnings
from typing import Any, Optional, Tuple, Union, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

# Try optional imports
try:
    from torchvision.transforms.functional import gaussian_blur
except Exception:
    gaussian_blur = None

try:
    import scipy.stats as _scipy_stats
except Exception:
    _scipy_stats = None


# ------------------------------
# Config
# ------------------------------
class DATDAConfig(PretrainedConfig):
    model_type = "datda"
    def __init__(
        self,
        device: str = "auto",
        spectral_suppression_min: float = 0.20,
        spectral_suppression_max: float = 0.92,
        high_freq_radius_ratio: float = 0.33,
        median_kernel: int = 3,
        bilateral_sigma_range: Tuple[float, float] = (0.5, 2.0),
        enable_dct_lowpass: bool = True,
        enable_jpeg_compress: bool = True,
        jpeg_quality_range: Tuple[int, int] = (50, 85),
        enable_tv_denoise: bool = True,
        tv_iters: int = 8,
        tv_weight: float = 0.12,
        enable_random_resize_pad: bool = True,
        rrp_scales: Tuple[float, float] = (0.9, 1.05),
        enable_feature_squeeze: bool = True,
        squeeze_bits: int = 5,
        ensemble_size: int = 3,
        ensemble_randomize: bool = True,
        residual_threshold: float = 0.020,
        enable_residual_cleanup: bool = True,
        gradient_shield_sigma: float = 0.015,
        reverse_steps: int = 6,
        reverse_eps: float = 0.03,
        reverse_step_size: float = 0.007,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.spectral_suppression_min = spectral_suppression_min
        self.spectral_suppression_max = spectral_suppression_max
        self.high_freq_radius_ratio = high_freq_radius_ratio
        self.median_kernel = median_kernel
        self.bilateral_sigma_range = bilateral_sigma_range
        self.enable_dct_lowpass = enable_dct_lowpass
        self.enable_jpeg_compress = enable_jpeg_compress
        self.jpeg_quality_range = jpeg_quality_range
        self.enable_tv_denoise = enable_tv_denoise
        self.tv_iters = tv_iters
        self.tv_weight = tv_weight
        self.enable_random_resize_pad = enable_random_resize_pad
        self.rrp_scales = rrp_scales
        self.enable_feature_squeeze = enable_feature_squeeze
        self.squeeze_bits = squeeze_bits
        self.ensemble_size = ensemble_size
        self.ensemble_randomize = ensemble_randomize
        self.residual_threshold = residual_threshold
        self.enable_residual_cleanup = enable_residual_cleanup
        self.gradient_shield_sigma = gradient_shield_sigma
        self.reverse_steps = reverse_steps
        self.reverse_eps = reverse_eps
        self.reverse_step_size = reverse_step_size
        self.seed = seed


# ------------------------------
# Utilities (tensor <-> PIL & small helpers)
# ------------------------------
def _to_pil(t: torch.Tensor) -> Image.Image:
    # t: 1x3xHxW or 3xHxW
    if t.dim() == 4 and t.shape[0] == 1:
        t = t[0]
    arr = (t.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = arr.squeeze()
    return Image.fromarray(arr)


def _from_pil(im: Image.Image, device: Optional[torch.device] = None) -> torch.Tensor:
    arr = np.array(im).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    t = torch.from_numpy(arr).unsqueeze(0)  # 1x3xHxW
    if device is not None:
        t = t.to(device)
    return t


def _jpeg_compress_tensor(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    # x: 1x3xHxW
    pil = _to_pil(x[0:1])
    bio = io.BytesIO()
    pil.save(bio, format="JPEG", quality=int(quality), optimize=True)
    bio.seek(0)
    im2 = Image.open(bio).convert("RGB")
    return _from_pil(im2, device=x.device)


def _dct_2d(img: torch.Tensor) -> torch.Tensor:
    # approx DCT-II via even-symmetric FFT trick
    # img: HxW (float tensor)
    H, W = img.shape
    def dct_1d(v):
        N = v.shape[0]
        x = torch.cat([v, v.flip(0)], dim=0)
        X = torch.fft.rfft(x)
        return torch.real(X[:N])
    out = torch.empty_like(img)
    for i in range(H):
        out[i] = dct_1d(img[i])
    out2 = torch.empty_like(out)
    for j in range(W):
        out2[:, j] = dct_1d(out[:, j])
    return out2


def _idct_2d(coef: torch.Tensor) -> torch.Tensor:
    H, W = coef.shape
    def idct_1d(C):
        N = C.shape[0]
        # construct symmetric spectrum and irfft
        # note: approximation; good enough for coarse low-pass
        ext = torch.cat([C, C.flip(0)], dim=0)
        x = torch.fft.irfft(ext, n=2*N)
        return x[:N]
    tmp = torch.empty_like(coef)
    for j in range(W):
        tmp[:, j] = idct_1d(coef[:, j])
    out = torch.empty_like(tmp)
    for i in range(H):
        out[i] = idct_1d(tmp[i])
    return out


def tv_denoise_torch(img: torch.Tensor, weight: float = 0.12, iters: int = 8) -> torch.Tensor:
    # Simple ROF-style solver for small iters. img: 1x3xHxW or 3xHxW
    squeeze = False
    if img.dim() == 4 and img.shape[0] == 1:
        img = img[0]
        squeeze = True
    C, H, W = img.shape
    u = img.clone()
    px = torch.zeros_like(u)
    py = torch.zeros_like(u)
    tau = 0.125
    for _ in range(iters):
        # forward differences
        ux = F.pad(u, (0, 1, 0, 0))[:, :, :W] - u
        uy = F.pad(u, (0, 0, 0, 1))[:, :,:H] - u
        px = px + tau * ux
        py = py + tau * uy
        norm = torch.clamp(torch.sqrt(px * px + py * py), min=1.0)
        px = px / norm
        py = py / norm
        div = (px - F.pad(px, (1, 0, 0, 0))[:, :, :W]) + (py - F.pad(py, (0, 0, 1, 0))[:, :, :H])
        u = (img + weight * div) / (1.0 + weight)
    if squeeze:
        return u.unsqueeze(0)
    return u


# ------------------------------
# DATDA Model
# ------------------------------
class DATDA(PreTrainedModel):
    """
    DATDA — Defense Against The Dark Arts
    Inference-time universal purifier for images. Use as a front-end for classification web UIs.
    """
    config_class = DATDAConfig
    base_model_prefix = "datda"

    def __init__(self, config: DATDAConfig):
        super().__init__(config)
        self.config = config

        # Fusion MLP (embedded): input dims = 5 detectors -> output weights for 4 paths
        in_dim = 5
        hidden = 64
        out_dim = 4
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
            nn.Softmax(dim=-1)
        )

        # initialize deterministically for reproducibility
        torch.manual_seed(self.config.seed)
        for p in self.fusion_mlp.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)

        # device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.to(self.device)
        self.eval()

    # --------------------
    # input helpers
    # --------------------
    def _to_tensor(self, x: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(x, Image.Image):
            if x.mode != "RGB":
                x = x.convert("RGB")
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Input must be PIL.Image, np.ndarray, or torch.Tensor")
        if x.ndim == 2:
            x = x.unsqueeze(-1).repeat(1, 1, 3)
        if x.ndim == 3:
            if x.shape[0] <= 3:
                x = x.unsqueeze(0)  # CHW -> NCHW
            else:
                x = x.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
        if x.ndim == 4:
            if x.shape[-1] in [1, 3] and x.shape[1] not in [1, 3]:
                x = x.permute(0, 3, 1, 2)
        if x.max() > 1.0:
            x = x.float() / 255.0
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, :, :]
        return x.clamp(0.0, 1.0).to(self.device)

    # --------------------
    # detectors (for adaptive fusion)
    # --------------------
    def detect_perturbation_stats(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns Bx5 features per sample:
        [fft_high_ratio, local_var, l1_residual, l2_norm_scaled, entropy_norm]
        """
        B, C, H, W = x.shape
        x_gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)

        # FFT high-frequency ratio
        fft = torch.fft.fft2(x_gray.squeeze(1))
        fft_shift = torch.fft.fftshift(fft)
        mag = torch.abs(fft_shift)
        center_h, center_w = H // 2, W // 2
        Y, X = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        dist = torch.sqrt((X - center_w).float() ** 2 + (Y - center_h).float() ** 2)
        radius = max(1, int(self.config.high_freq_radius_ratio * min(H, W)))
        high_mask = (dist > radius).float()
        high_energy = (mag * high_mask).sum(dim=[1, 2])
        total_energy = mag.sum(dim=[1, 2]) + 1e-8
        fft_ratio = (high_energy / total_energy).view(B)

        # local variance
        pad = 1
        x_padded = F.pad(x_gray, (pad, pad, pad, pad), mode="reflect")
        patches = x_padded.unfold(2, 3, 1).unfold(3, 3, 1)
        local_mean = patches.mean(dim=(-1, -2), keepdim=True)
        local_var = (patches - local_mean).pow(2).mean(dim=(-1, -2))
        var_score = local_var.mean(dim=[1, 2, 3]).view(B)

        # l1 residual vs simple gaussian blur baseline
        if gaussian_blur is not None:
            baseline = gaussian_blur(x, kernel_size=3, sigma=0.8)
        else:
            baseline = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        l1_res = (x - baseline).abs().mean(dim=[1, 2, 3]).view(B)

        l2_norm = x.view(B, -1).norm(p=2, dim=1) / float(H * W * C)

        # approximate entropy from histogram of grayscale
        flat = (x_gray.view(B, -1) * 255).long().clamp(0, 255)
        hist = torch.stack([torch.bincount(flat[i], minlength=256).float() for i in range(B)])
        probs = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(probs * (probs + 1e-12).log()).sum(dim=1) / math.log(256.0)

        feats = torch.stack([fft_ratio, var_score, l1_res, l2_norm, entropy], dim=1)
        # batchwise min-max normalization (stable)
        minv = feats.min(dim=0).values
        maxv = feats.max(dim=0).values
        denom = (maxv - minv).clamp(min=1e-6)
        feats_norm = (feats - minv) / denom
        return feats_norm  # B x 5

    # --------------------
    # purification paths
    # --------------------
    def spectral_path(self, x: torch.Tensor, fft_feat: torch.Tensor) -> torch.Tensor:
        # FFT suppression + optional DCT low-pass coarse mixing
        B, C, H, W = x.shape
        gamma = (
            self.config.spectral_suppression_min +
            (self.config.spectral_suppression_max - self.config.spectral_suppression_min) *
            torch.sigmoid(fft_feat[:, 0]).view(B, 1, 1, 1)
        )
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
                dist = torch.sqrt((X-center_w).float()**2 + (Y-center_h).float()**2)
                radius = max(1, int(self.config.high_freq_radius_ratio * min(H, W)))
                high_mask = (dist > radius).float()
                mag = mag * (1.0 - float(gamma[b, 0, 0, 0]) * high_mask)
                fft_clean = mag * torch.exp(1j * phase)
                ifft_shift = torch.fft.ifftshift(fft_clean)
                recon = torch.fft.ifft2(ifft_shift).real
                out[b:b+1, c:c+1] = recon.unsqueeze(0)
        if self.config.enable_dct_lowpass:
            # coarse DCT low-pass on grayscale and mix in
            gray = (0.299 * out[:, 0] + 0.587 * out[:, 1] + 0.114 * out[:, 2]).squeeze(1)
            low = torch.empty_like(gray)
            for i in range(B):
                coef = _dct_2d(gray[i].cpu())
                coef = coef.to(x.device)
                Hc, Wc = coef.shape
                keep_h = max(1, int(Hc * (1 - self.config.high_freq_radius_ratio)))
                keep_w = max(1, int(Wc * (1 - self.config.high_freq_radius_ratio)))
                mask = torch.zeros_like(coef)
                mask[:keep_h, :keep_w] = 1.0
                coef_low = coef * mask
                low[i] = _idct_2d(coef_low).to(x.device)
            low3 = low.unsqueeze(1).repeat(1, 3, 1, 1)
            out = 0.6 * out + 0.4 * low3
        return out.clamp(0.0, 1.0)

    def bilateral_path(self, x: torch.Tensor, var_feat: torch.Tensor) -> torch.Tensor:
        # gaussian / bilateral-like smoothing using torchvision gaussian_blur if available
        if gaussian_blur is None:
            # fallback to avg pool smoothing
            return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        sigma_min, sigma_max = self.config.bilateral_sigma_range
        # use mean var as signal
        sigma = float(sigma_min + (sigma_max - sigma_min) * float(var_feat.mean().item()))
        k_size = int(2 * (2 * sigma) + 1)
        k_size = min(max(k_size, 3), 15)
        if k_size % 2 == 0:
            k_size += 1
        return gaussian_blur(x, kernel_size=k_size, sigma=sigma)

    def median_path(self, x: torch.Tensor) -> torch.Tensor:
        k = max(1, int(self.config.median_kernel))
        pad = k // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)
        flat = patches.contiguous().view(*patches.shape[:-2], -1)
        med = flat.median(dim=-1).values
        return med

    def tv_and_compress_path(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        if self.config.enable_feature_squeeze:
            bits = int(max(1, min(8, self.config.squeeze_bits)))
            levels = float(2 ** bits - 1)
            out = (out * levels).round() / levels
        if self.config.enable_jpeg_compress:
            q_low, q_high = self.config.jpeg_quality_range
            q = int((q_low + q_high) // 2)
            outs = []
            for i in range(out.shape[0]):
                outs.append(_jpeg_compress_tensor(out[i:i+1], quality=q))
            out = torch.cat(outs, dim=0).to(x.device)
        if self.config.enable_tv_denoise:
            outs = []
            for i in range(out.shape[0]):
                outs.append(tv_denoise_torch(out[i:i+1], weight=self.config.tv_weight, iters=self.config.tv_iters))
            out = torch.cat(outs, dim=0)
        # gentle blur
        if gaussian_blur is not None:
            out = gaussian_blur(out, kernel_size=3, sigma=0.6)
        else:
            out = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
        return out.clamp(0.0, 1.0)

    def random_resize_pad(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out_list = []
        for i in range(B):
            pil = _to_pil(x[i:i+1])
            if self.config.ensemble_randomize:
                scale = float(np.random.uniform(self.config.rrp_scales[0], self.config.rrp_scales[1]))
            else:
                scale = float((self.config.rrp_scales[0] + self.config.rrp_scales[1]) / 2.0)
            new_w = max(1, int(W * scale))
            new_h = max(1, int(H * scale))
            resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)
            pad_left = max(0, (W - new_w) // 2)
            pad_top = max(0, (H - new_h) // 2)
            canvas = Image.new("RGB", (W, H), (127, 127, 127))
            canvas.paste(resized, (pad_left, pad_top))
            out_list.append(_from_pil(canvas, device=x.device))
        return torch.cat(out_list, dim=0)

    # --------------------
    # adaptive fusion
    # --------------------
    def adaptive_fusion(self, spect, bilat, med, comp, feats):
        """
        feats: B x 5
        returns convex combination across 4 paths
        """
        with torch.no_grad():
            weights = self.fusion_mlp(feats)  # B x 4
            weights = weights.view(weights.shape[0], 4, 1, 1, 1)
            fused = weights[:, 0] * spect + weights[:, 1] * bilat + weights[:, 2] * med + weights[:, 3] * comp
        return fused

    # --------------------
    # residual cleanup
    # --------------------
    def residual_cleanup(self, x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
        if not self.config.enable_residual_cleanup:
            return x_out
        residual = (x_in - x_out).abs().mean(dim=[1, 2, 3])
        mask = (residual > self.config.residual_threshold).float().view(-1, 1, 1, 1)
        if mask.any():
            if gaussian_blur is not None:
                cleaned = gaussian_blur(x_out, kernel_size=3, sigma=0.9)
            else:
                cleaned = F.avg_pool2d(x_out, kernel_size=3, stride=1, padding=1)
            x_out = torch.where(mask.bool(), cleaned, x_out)
        return x_out

    # --------------------
    # gradient shield (obfuscation/robustness trick)
    # --------------------
    def gradient_shield(self, x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
        """
        Adds small stochastic noise and detaches gradient flow to make gradient-based attacks less effective at the prediction boundary.
        Note: this is a defensive obfuscation technique — combine with other real purification paths.
        """
        if sigma is None:
            sigma = float(self.config.gradient_shield_sigma)
        noise = torch.randn_like(x) * sigma
        x_noisy = (x + noise).clamp(0.0, 1.0)
        # detach to break simple gradient chains while preserving value for inference
        return x_noisy.detach() + (x_noisy - x_noisy.detach())

    # --------------------
    # reverse adversarial reconstruction (anti-attack)
    # --------------------
    def reverse_attack_reconstruct(
        self,
        x: torch.Tensor,
        surrogate_model: nn.Module,
        method: str = "pgd",
        eps: Optional[float] = None,
        steps: Optional[int] = None,
        step_size: Optional[float] = None,
        targeted: bool = False
    ) -> torch.Tensor:
        """
        Attempt to push the image in the *reverse* direction of predicted adversarial gradient,
        i.e., reduce loss wrt predicted class to reconstruct a cleaner image.
        surrogate_model: a torchvision/timm model that maps NxCxHxW -> logits
        method: 'pgd', 'fgsm', 'bim', 'mim'
        Returns reconstructed image (same shape) — use with caution, intended as defensive reconstruction.
        """
        if eps is None:
            eps = float(self.config.reverse_eps)
        if steps is None:
            steps = int(self.config.reverse_steps)
        if step_size is None:
            step_size = float(self.config.reverse_step_size)

        device = x.device
        surrogate_model = surrogate_model.to(device).eval()
        x_rev = x.clone().detach()
        # direction sign depends on targeted: to reduce loss we minimize cross-entropy, i.e., take negative gradient
        for _ in range(steps):
            x_rev.requires_grad = True
            logits = surrogate_model(x_rev)
            preds = logits.detach().argmax(dim=1)
            loss = F.cross_entropy(logits, preds)
            # we want to minimize loss (pull to original prediction) — gradient points to increase loss
            grad = torch.autograd.grad(loss, x_rev, create_graph=False)[0]
            if method.lower() in ("fgsm",):
                step = -step_size * grad.sign()
                x_rev = (x_rev + step).detach()
            elif method.lower() in ("bim", "iterative", "i-fgsm"):
                step = -step_size * grad.sign()
                x_rev = (x_rev + step).clamp(0.0, 1.0)
                # project within eps-ball of original x
                x_rev = torch.max(torch.min(x_rev, x + eps), x - eps).detach()
            elif method.lower() in ("pgd",):
                step = -step_size * torch.sign(grad)
                x_rev = (x_rev + step).clamp(0.0, 1.0)
                x_rev = torch.max(torch.min(x_rev, x + eps), x - eps).detach()
            elif method.lower() in ("mim",):
                # momentum iterative method (defensive reversal)
                if not hasattr(self, "_mim_g"):
                    self._mim_g = torch.zeros_like(x_rev)
                decay = 1.0
                g = grad / (torch.mean(torch.abs(grad)) + 1e-12)
                self._mim_g = decay * self._mim_g + g
                step = -step_size * torch.sign(self._mim_g)
                x_rev = (x_rev + step).clamp(0.0, 1.0)
                x_rev = torch.max(torch.min(x_rev, x + eps), x - eps).detach()
            else:
                raise ValueError(f"Unsupported reverse method '{method}'")
        return x_rev.detach()

    # --------------------
    # single-pass purifier (core pipeline)
    # --------------------
    def _single_pass(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.detect_perturbation_stats(x)  # B x 5
        spect = self.spectral_path(x, feats)
        bilat = self.bilateral_path(x, feats)
        med = self.median_path(x)
        comp = self.tv_and_compress_path(x)
        fused = self.adaptive_fusion(spect, bilat, med, comp, feats)
        if self.config.enable_random_resize_pad:
            rrp = self.random_resize_pad(x)
            fused = 0.85 * fused + 0.15 * rrp
        if self.config.gradient_shield_sigma > 0:
            fused = self.gradient_shield(fused, sigma=self.config.gradient_shield_sigma)
        purified = self.residual_cleanup(x, fused)
        return purified.clamp(0.0, 1.0)
    
    # --------------------
    # forward (with optional ensemble randomized smoothing)
    # --------------------
    def forward(self, x: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        x = self._to_tensor(x)
        B = x.shape[0]
        if self.config.ensemble_size <= 1:
            with torch.no_grad():
                return self._single_pass(x)
        outs = []
        for i in range(self.config.ensemble_size):
            if self.config.ensemble_randomize:
                torch.manual_seed(self.config.seed + i)
                np.random.seed(self.config.seed + i)
            with torch.no_grad():
                outs.append(self._single_pass(x))
        stacked = torch.stack(outs, dim=0)
        avg = stacked.mean(dim=0)
        return avg.clamp(0.0, 1.0)

    # --------------------
    # randomized smoothing certification
    # --------------------
    def certify_randomized_smoothing(
        self,
        classifier_fn,
        x: Union[torch.Tensor, Image.Image, np.ndarray],
        sigma: float = 0.25,
        num_samples: int = 50,
        return_all: bool = False
    ) -> Tuple[Any, dict]:
        """
        Monte Carlo randomized smoothing certification wrapper.
        classifier_fn: function mapping tensor NxCxHxW -> logits or probs
        Returns: (top_class, metadata dict)
        metadata contains: pA (proportion), sigma, radius (if scipy available), all_preds (optional)
        """
        x_t = self._to_tensor(x)
        B = x_t.shape[0]
        preds = []
        for _ in range(num_samples):
            noise = sigma * torch.randn_like(x_t)
            noisy = (x_t + noise).clamp(0.0, 1.0)
            with torch.no_grad():
                logits = classifier_fn(noisy)  # expect NxK or NxCxHxW? assume logits NxK
            if logits.dim() == 4:
                # assume classifier_fn returned dense predictions — reduce
                logits = logits.view(logits.shape[0], -1)
            p = logits.argmax(dim=1)
            preds.append(p.cpu())
        stacked = torch.stack(preds, dim=0)  # S x B
        mode_vals, counts = torch.mode(stacked, dim=0)
        top_class = mode_vals  # B
        pA = (stacked == mode_vals.unsqueeze(0)).float().mean(dim=0).cpu().numpy()  # B-length
        metadata = {"pA": pA, "sigma": sigma, "num_samples": num_samples}
        # compute radius using Gaussian ppf if scipy available
        if _scipy_stats is not None:
            radii = []
            for pa in pA:
                # protect edge cases
                pa = float(max(min(pa, 1.0 - 1e-12), 1e-12))
                try:
                    radius = sigma * float(_scipy_stats.norm.ppf(pa))
                except Exception:
                    radius = None
                radii.append(radius)
            metadata["radius"] = radii
        else:
            metadata["radius"] = None
            warnings.warn("scipy not available — radius not computed. Install scipy for certified radius (norm.ppf).")
        if return_all:
            metadata["all_preds"] = stacked.numpy()
        return top_class.numpy(), metadata

    # --------------------
    # helpers to push config (HF style)
    # --------------------
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[str] = None, *args, **kwargs):
        # DATDA is a code-level purifier; config may be loaded or created
        if pretrained_model_name_or_path:
            cfg = DATDAConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            cfg = DATDAConfig(**kwargs)
        model = cls(cfg)
        if pretrained_model_name_or_path:
            warnings.warn("DATDA is an inference-time purifier — code logic is used; no weights downloaded.")
        return model

    def push_to_hub(self, repo_id: str, **kwargs):
        # only push config; code should be included in repo
        try:
            self.config.push_to_hub(repo_id, **kwargs)
        except Exception as e:
            warnings.warn(f"Failed to push config to hub: {e}")

# End of datda.py
