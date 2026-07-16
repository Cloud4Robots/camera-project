"""
depth_augmentations.py
=======================
Phase 1 augmentations for the RGB-guided IR depth correction model.
Trains on RAW (filters-off) D415 data per team decision (2026-07-09).

Three steps, run in order via augment():
  1. geometric_augmentation  - same flip/rotate applied to rgb, ir, gt together
  2. photometric_augmentation - brightness/contrast jitter, RGB only
  3. inject_noise             - corrupts ir to look like raw sensor output,
                                 based on real D415 testing (IMG_1461/1462):
                                   - random speckle dropout, worse on dark surfaces
                                   - thin structures (cables/rods) fragmenting
                                   - flat, low-texture surfaces dropping out entirely
                                     (the "shell disappears" finding)
"""

import numpy as np
import cv2


def geometric_augmentation(rgb, ir, gt, rng=None):
    """Flip + rotate rgb/ir/gt with the SAME transform so they stay aligned."""
    rng = rng or np.random.default_rng()
    h, w = gt.shape[:2]

    flip = rng.random() < 0.5
    angle = rng.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    def warp(img, interp):
        out = cv2.warpAffine(img, M, (w, h), flags=interp)
        return cv2.flip(out, 1) if flip else out

    rgb_out = warp(rgb, cv2.INTER_LINEAR)
    ir_out = warp(ir, cv2.INTER_NEAREST)   # nearest: don't invent fake depth at edges
    gt_out = warp(gt, cv2.INTER_NEAREST)
    return rgb_out, ir_out, gt_out


def photometric_augmentation(rgb, rng=None):
    """Random brightness/contrast jitter. RGB only - never touches ir/gt."""
    rng = rng or np.random.default_rng()
    out = rgb.astype(np.float32)
    if out.max() > 1.5:
        out = out / 255.0

    brightness = rng.uniform(0.8, 1.2)
    contrast = rng.uniform(0.8, 1.2)
    out = out * brightness
    out = (out - out.mean()) * contrast + out.mean()
    return np.clip(out, 0, 1)


def inject_noise(ir, gt, rgb, rng=None, severity=0.15):
    """
    Corrupt `ir` to look like raw (unfiltered) D455 output. Three failure
    types, based on real hardware testing:
      - speckle: random dropout everywhere, worse on dark surfaces
      - thin structures (cables/rods): fragment/disappear
      - flat + low-texture surfaces: drop out almost entirely (raw stereo
        matching has nothing to correlate on - this is why a flat shell
        piece "disappears" instead of just getting a smoothed depth value)
    """
    rng = rng or np.random.default_rng()
    ir_out = ir.copy().astype(np.float32)
    h, w = gt.shape[:2]
    gt_filled = np.nan_to_num(gt)

    gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    if gray.max() > 1.5:
        gray = gray / 255.0
    darkness = 1.0 - gray  # 0 = bright, 1 = dark; dark surfaces fail more

    # 1. speckle noise everywhere, worse on dark surfaces
    speckle = rng.random((h, w)) < severity * (0.5 + darkness)
    ir_out[speckle] = 0

    # 2. thin structures: "object" = depth clearly closer than the local
    # average (something sticking out toward the camera); "thin" = it
    # vanishes when eroded by a small kernel
    local_avg = cv2.blur(gt_filled, (31, 31))
    is_object = (local_avg - gt_filled) > 0.05
    eroded = cv2.erode(is_object.astype(np.uint8), np.ones((6, 6), np.uint8))
    is_thin = is_object & (eroded == 0)
    frag = is_thin & (rng.random((h, w)) < severity * 1.5)
    ir_out[frag] = 0

    # 3. flat + low-texture surfaces: little RGB gradient + little depth
    # gradient = raw stereo matching fails here almost entirely
    gy, gx = np.gradient(gt_filled)
    is_flat = (np.abs(gy) + np.abs(gx)) < 0.01
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    texture = cv2.blur(np.abs(sobel_x) + np.abs(sobel_y), (9, 9))
    is_low_texture = texture < 0.02
    dropout = is_flat & is_low_texture & (rng.random((h, w)) < severity * 2.5)
    ir_out[dropout] = 0

    return ir_out


def augment(rgb, ir, gt, rng=None):
    """Run all three steps in order: geometric -> photometric -> noise."""
    rng = rng or np.random.default_rng()
    rgb, ir, gt = geometric_augmentation(rgb, ir, gt, rng)
    rgb = photometric_augmentation(rgb, rng)
    ir = inject_noise(ir, gt, rgb, rng)
    return rgb, ir, gt


if __name__ == "__main__":
    # smoke test: a thin "rod" + a flat, low-texture "shell" patch
    rng = np.random.default_rng(0)
    h, w = 128, 128
    rgb = rng.uniform(0, 1, (h, w, 3)).astype(np.float32)
    rgb[100:, :] = 0.6  # flat, low-texture patch

    gt = np.full((h, w), 2.0, dtype=np.float32)
    gt[100:, :] = 2.5     # flat region (matches the low-texture patch above)
    gt[40:88, 60:64] = 1.0  # thin vertical rod
    ir = gt.copy()

    rgb2, ir2, gt2 = augment(rgb, ir, gt, rng)
    print("rgb:", rgb2.shape, "ir:", ir2.shape, "gt:", gt2.shape)
    print("holes after augmentation:", int((ir2 == 0).sum()), "/", ir2.size)
