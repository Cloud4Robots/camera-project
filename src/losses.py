"""
losses.py
=========
Composite loss for RGB-guided IR depth correction training:

    L_total = alpha * L_pixel(L1) + beta * L_grad + gamma * L_ssim

Why three terms, together (not just L1):
- L1 pixel loss alone tends to produce blurry, averaged-out depth - it
  penalizes magnitude of error everywhere equally, so the easiest way for
  the network to minimize it is to smear values toward a safe average.
- Gradient loss compares Sobel gradients of pred vs. ground truth, which
  directly penalizes blurred/broken edges - this targets the "thin
  structures (cables/rods) fragmenting" failure mode from
  depth_augmentations.py.
- SSIM loss compares local structural similarity (mean/variance/covariance
  in a sliding window) rather than raw pixel values, which is sensitive to
  whether a whole region's structure is intact - this targets the "flat,
  low-texture surfaces dropping out entirely" failure mode.

Initial weights: alpha=1.0, beta=0.5, gamma=0.5.
This follows the common convention in monocular/guided depth literature
(e.g. DenseDepth-style setups): pixel loss anchors absolute depth values
and dominates early training; gradient + SSIM are secondary "sharpening /
structure" terms weighted lower so they refine rather than dominate the
signal. These are starting points, not tuned values - re-check once
Phase 1 loss curves come in (if grad/ssim barely move the total, raise
beta/gamma; if training destabilizes, lower them).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """L1 distance between Sobel gradients of pred and target depth maps."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradients(self, x: torch.Tensor):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_gx, pred_gy = self._gradients(pred)
        target_gx, target_gy = self._gradients(target)
        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


class SSIMLoss(nn.Module):
    """
    1 - SSIM, single-channel (depth), computed with a Gaussian window.
    No external dependency (no pytorch-msssim needed) - small enough to
    just implement directly.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 c1: float = 0.01 ** 2, c2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2

        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(0)
        window_2d = g.t() @ g  # outer product -> 2D Gaussian kernel
        self.register_buffer("window", window_2d.view(1, 1, window_size, window_size))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        window = self.window

        mu_pred = F.conv2d(pred, window, padding=pad)
        mu_target = F.conv2d(target, window, padding=pad)
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, window, padding=pad) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=pad) - mu_pred_target

        numerator = (2 * mu_pred_target + self.c1) * (2 * sigma_pred_target + self.c2)
        denominator = (mu_pred_sq + mu_target_sq + self.c1) * (sigma_pred_sq + sigma_target_sq + self.c2)
        ssim_map = numerator / denominator

        return 1.0 - ssim_map.mean()


class DepthCorrectionLoss(nn.Module):
    """
    Combined training loss: L_total = alpha*L1 + beta*Gradient + gamma*SSIM.
    Returns (total_loss_tensor, dict_of_component_floats) so callers can
    log each term separately during training.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.grad_loss = GradientLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l_pixel = F.l1_loss(pred, target)
        l_grad = self.grad_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)

        total = self.alpha * l_pixel + self.beta * l_grad + self.gamma * l_ssim

        components = {
            "pixel": l_pixel.item(),
            "grad": l_grad.item(),
            "ssim": l_ssim.item(),
            "total": total.item(),
        }
        return total, components


if __name__ == "__main__":
    # Quick isolated check: each term is finite and >= 0 on random data,
    # and drops to ~0 when pred == target.
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)

    criterion = DepthCorrectionLoss()
    total, parts = criterion(pred, target)
    print("random pred vs target:", parts)
    assert torch.isfinite(total), "loss is not finite"

    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "bad gradients"
    print("backward OK, grad finite")

    same = target.clone().requires_grad_(True)
    total_same, parts_same = criterion(same, target)
    print("pred == target:", parts_same)
    assert parts_same["pixel"] < 1e-6
    assert parts_same["grad"] < 1e-6
    assert parts_same["ssim"] < 1e-4  # SSIM has small numerical slack near 0

    print("\nlosses.py self-check PASSED")
