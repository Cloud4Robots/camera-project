"""
synthetic_dataset.py — procedurally generates synthetic (rgb, ir, gt)
training triples for Phase 1 pretraining (no real captured data needed).

Thin-structure sizing is calibrated against Rhys's 7/15 real D415 failure
testing (single D415, default preset, 848x480 test / 1280x720 recording,
measured at 20-25in / ~510-635mm):
  - failure is geometric, not material (shiny plastic mostly fine)
  - threshold is ~3-8mm depending on color at that distance:
      white  ~7mm pen holder -> holds shape, rough edges
      5/16in (~7.9mm) rod    -> solid
      ~3mm black cable       -> fragments completely
  - darker surfaces measurably worse than lighter ones at the same diameter
  - threshold gets WORSE (structures need to be thicker to survive) at
    longer range, since apparent pixel width shrinks with distance
  - edges are rough everywhere, more so as features get thinner
  - flat objects lying flush with the background/floor nearly vanish in
    depth even though RGB sees them clearly ("ground-blending")
Camera intrinsics (fx) are from extract_intrinsics.py's real D415 readout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from depth_augmentations import augment

# ─── Real-camera calibration constants (from Rhys's 7/15 D415 testing + extract_intrinsics.py) ──
CAMERA_FX_PX = 900.0            # measured D415 fx at 1280x720 (899-925 across units)
TEST_DISTANCE_MM_RANGE = (510, 635)   # Rhys's 20-25in test range, in mm

# Real-world thin-structure diameters that map to "survives" vs "fragments",
# per Rhys's measurements. Sampling across this range (not just the failure
# point) so training sees both "just barely survives" and "fully fragments"
# cases, matching what's actually been observed.
ROD_DIAMETER_MM_RANGE = (3.0, 8.0)     # 3mm cable (fragments) .. ~8mm rod (solid)
FRAGMENT_DIAMETER_MM = 4.0             # below this, treat as "should fragment more" (dark cable regime)

BG_DEPTH_RANGE = (2.0, 3.0)       # meters, general D415 usable range
OBJECT_DEPTH_RANGE = (0.5, 1.8)
ROD_DEPTH_RANGE = (0.4, 1.0)


def _mm_to_px(diameter_mm: float, img_w_px: int, native_w_px: int = 1280,
              distance_mm: float | None = None, rng: np.random.Generator | None = None) -> int:
    """
    Convert a real-world diameter (mm) to a pixel width in an `img_w_px`-wide
    synthetic image, using the real D415 fx and a sampled test distance.
    Farther distance -> smaller apparent pixel width -> matches Rhys's note
    that the failure threshold gets worse at longer range.
    """
    if distance_mm is None:
        distance_mm = rng.uniform(*TEST_DISTANCE_MM_RANGE)
    px_at_native_res = CAMERA_FX_PX * diameter_mm / distance_mm
    px_scaled = px_at_native_res * (img_w_px / native_w_px)
    return max(1, round(px_scaled))


def _paint(gt, rgb, y0, y1, x0, x1, depth, color):
    gt[y0:y1, x0:x1] = depth
    rgb[y0:y1, x0:x1] = color


def _draw_blob(gt, rgb, rng, h, w):
    """Rectangular object, closer than background, random color."""
    bh, bw = rng.integers(h // 8, h // 3), rng.integers(w // 8, w // 3)
    y0, x0 = rng.integers(0, h - bh), rng.integers(0, w - bw)
    _paint(gt, rgb, y0, y0 + bh, x0, x0 + bw,
           rng.uniform(*OBJECT_DEPTH_RANGE), rng.uniform(0.1, 0.9, size=3))


def _draw_rod(gt, rgb, rng, h, w):
    """
    Thin rod, closer than background - gradient-loss target.
    Diameter sampled in real mm (3-8mm, Rhys's measured survive/fragment
    range) then converted to pixels via real camera fx + sampled distance.
    Darker color is biased for thinner rods, matching the observed
    "dark + thin fragments, light + thick survives" pattern.
    """
    diameter_mm = rng.uniform(*ROD_DIAMETER_MM_RANGE)
    thickness = _mm_to_px(diameter_mm, w, rng=rng)
    depth = rng.uniform(*ROD_DEPTH_RANGE)

    # thin (near 3mm, cable-like) -> biased dark; thick (near 8mm, rod-like) -> lighter allowed
    dark_bias = np.clip((FRAGMENT_DIAMETER_MM * 2 - diameter_mm) / (FRAGMENT_DIAMETER_MM * 2), 0, 1)
    color_max = 0.3 + 0.5 * (1 - dark_bias)  # thin cables stay dark; thicker rods can be lighter
    color = rng.uniform(0.0, color_max, size=3)

    if rng.random() < 0.5:  # vertical
        x0 = rng.integers(w // 6, 5 * w // 6)
        _paint(gt, rgb, 0, h, x0, x0 + thickness, depth, color)
    else:  # horizontal
        y0 = rng.integers(h // 6, 5 * h // 6)
        _paint(gt, rgb, y0, y0 + thickness, 0, w, depth, color)


def _draw_flat_patch(gt, rgb, rng, h, w):
    """
    Flat, low-texture region placed near background depth - SSIM-loss
    target ("ground-blending": Rhys confirmed a flat object lying flush
    with the background nearly disappears in depth despite being clearly
    visible in RGB). Depth offset kept small to reproduce that near-flush case.
    """
    ph, pw = rng.integers(h // 4, h // 2), rng.integers(w // 4, w // 2)
    y0, x0 = rng.integers(0, h - ph), rng.integers(0, w - pw)
    depth = rng.uniform(*BG_DEPTH_RANGE) + rng.uniform(0.02, 0.15)  # near-flush, not a big step
    color = np.clip(rng.uniform(0.4, 0.7) + rng.normal(0, 0.01, size=3), 0, 1)
    _paint(gt, rgb, y0, y0 + ph, x0, x0 + pw, depth, color)


def generate_scene(h: int, w: int, rng: np.random.Generator):
    """Build one clean (rgb, gt) synthetic scene, pre-augmentation/corruption."""
    gt = np.full((h, w), rng.uniform(*BG_DEPTH_RANGE), dtype=np.float32)
    rgb = np.tile(rng.uniform(0.3, 0.6, size=3), (h, w, 1)).astype(np.float32)
    rgb += rng.normal(0, 0.02, size=rgb.shape).astype(np.float32)  # faint bg texture

    for _ in range(rng.integers(2, 5)):
        _draw_blob(gt, rgb, rng, h, w)
    for _ in range(rng.integers(1, 3)):
        _draw_rod(gt, rgb, rng, h, w)
    _draw_flat_patch(gt, rgb, rng, h, w)

    return np.clip(rgb, 0, 1).astype(np.float32), gt


class SyntheticDepthDataset(Dataset):
    """
    On-the-fly synthetic dataset. `length` is a nominal epoch size - there's
    no fixed underlying data; each index deterministically seeds a unique
    random scene (base_seed + idx), so results are reproducible across runs.
    """

    def __init__(self, length: int = 500, h: int = 128, w: int = 128, base_seed: int = 0):
        self.length, self.h, self.w, self.base_seed = length, h, w, base_seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + idx)
        rgb, gt = generate_scene(self.h, self.w, rng)
        rgb_aug, ir_aug, gt_aug = augment(rgb, gt.copy(), gt, rng)  # ir starts == gt, pre-corruption

        return (torch.from_numpy(rgb_aug).permute(2, 0, 1).float(),   # (3,H,W)
                torch.from_numpy(ir_aug).unsqueeze(0).float(),         # (1,H,W)
                torch.from_numpy(gt_aug).unsqueeze(0).float())         # (1,H,W)


if __name__ == "__main__":
    ds = SyntheticDepthDataset(length=8, h=128, w=128, base_seed=0)
    print(f"dataset length: {len(ds)}")

    # sanity check: print the real-mm -> px conversion range at this IMG_SIZE
    rng0 = np.random.default_rng(0)
    px_thin = _mm_to_px(ROD_DIAMETER_MM_RANGE[0], 128, distance_mm=TEST_DISTANCE_MM_RANGE[1])
    px_thick = _mm_to_px(ROD_DIAMETER_MM_RANGE[1], 128, distance_mm=TEST_DISTANCE_MM_RANGE[0])
    print(f"rod thickness range at 128px: {px_thin}-{px_thick} px "
          f"(from Rhys's {ROD_DIAMETER_MM_RANGE[0]}-{ROD_DIAMETER_MM_RANGE[1]}mm real threshold)")

    for i in range(3):
        rgb, ir, gt = ds[i]
        print(f"sample {i}: rgb {tuple(rgb.shape)}  ir {tuple(ir.shape)}  gt {tuple(gt.shape)}  "
              f"gt range [{gt.min():.2f}, {gt.max():.2f}]  "
              f"ir zeros (dropout): {int((ir == 0).sum())}/{ir.numel()}")
    print("\nsynthetic_dataset.py smoke test PASSED")