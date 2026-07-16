"""
sanity_check_loss.py
=====================
Today's sanity check: confirm the composite loss (L1 + gradient + SSIM)
works correctly with DepthCorrectionNet - not full training, just:

  1. Forward pass -> loss -> backward, no errors, all finite numbers
  2. Overfit test: on ONE tiny fake (rgb, ir, gt) triple, loss should
     drop sharply over ~100 steps. If it doesn't, either the loss or the
     model has a wiring bug - better to find that out now than after a
     multi-hour Phase 1 run tomorrow.

Uses depth_augmentations.py's fake rod+shell sample as the ground truth,
and derives a noisy `ir` input the same way its own smoke test does.

Usage:
    python sanity_check_loss.py
"""

import numpy as np
import torch
import torch.optim as optim

from depth_correction_model import DepthCorrectionNet
from losses import DepthCorrectionLoss
from depth_augmentations import inject_noise


def make_fake_sample(h=64, w=64, seed=0):
    """Same synthetic rod + flat-shell pattern as depth_augmentations.py's
    own smoke test, just smaller so this runs fast."""
    rng = np.random.default_rng(seed)

    rgb = rng.uniform(0, 1, (h, w, 3)).astype(np.float32)
    rgb[h // 2:, :] = 0.6  # flat, low-texture patch

    gt = np.full((h, w), 2.0, dtype=np.float32)
    gt[h // 2:, :] = 2.5                      # flat region
    gt[h // 4:h // 2, w // 3:w // 3 + 4] = 1.0  # thin vertical rod

    ir = inject_noise(gt.copy(), gt, rgb, rng=rng, severity=0.15)
    return rgb, ir, gt


def to_tensors(rgb, ir, gt):
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()   # (1,3,H,W)
    ir_t = torch.from_numpy(ir).unsqueeze(0).unsqueeze(0).float()          # (1,1,H,W)
    gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float()          # (1,1,H,W)
    return rgb_t, ir_t, gt_t


def main():
    torch.manual_seed(0)
    rgb, ir, gt = make_fake_sample()
    rgb_t, ir_t, gt_t = to_tensors(rgb, ir, gt)

    model = DepthCorrectionNet(base_channels=32, num_heads=4)
    criterion = DepthCorrectionLoss(alpha=1.0, beta=0.5, gamma=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("--- step 0: single forward/backward, check nothing explodes ---")
    pred = model(rgb_t, ir_t)
    total, parts = criterion(pred, gt_t)
    assert torch.isfinite(total), "loss is not finite on first forward pass"
    total.backward()
    print(f"  pred shape: {tuple(pred.shape)}  gt shape: {tuple(gt_t.shape)}")
    print(f"  loss components: {parts}")
    optimizer.zero_grad()

    print("\n--- overfitting on 1 fake sample for 150 steps ---")
    losses = []
    for step in range(150):
        optimizer.zero_grad()
        pred = model(rgb_t, ir_t)
        total, parts = criterion(pred, gt_t)
        total.backward()
        optimizer.step()
        losses.append(parts["total"])
        if step % 25 == 0 or step == 149:
            print(f"  step {step:3d}  total={parts['total']:.4f}  "
                  f"pixel={parts['pixel']:.4f}  grad={parts['grad']:.4f}  ssim={parts['ssim']:.4f}")

    print(f"\nfirst loss: {losses[0]:.4f}   last loss: {losses[-1]:.4f}")
    improved = losses[-1] < losses[0] * 0.5
    print(f"[{'PASS' if improved else 'FAIL'}] loss dropped by >50% over 150 steps "
          f"(overfit sanity check {'passed' if improved else 'FAILED - check wiring'})")


if __name__ == "__main__":
    main()
