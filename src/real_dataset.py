"""
real_dataset.py — loads timestamp-aligned (rgb, intel_depth, ffs_depth)
triples from the real FFS capture for Phase 2 fine-tuning.

Expects `root` laid out as:
    root/rgb/left_{ts}.png
    root/intel_depth/{ts}.npy   <- raw IR depth (noisy input)
    root/ffs_depth/{ts}.npy     <- stereo-matched ground truth

Handles two things the notebook only checked ad hoc:
- clips intel_depth outliers (>3m is sensor noise, not real returns)
- crops/resizes 1280x720 -> 640x480 to match ground truth, using nearest
  neighbor for depth (no invented edge values) and linear for RGB (quality)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# if this is over 3 meters depth -- it is considered as a noise
MAX_VALID_DEPTH_M = 3.0

# cut the RGB and Ground Truth so that they are in the same size 
def crop_720p_to_4_3(img: np.ndarray, is_depth: bool) -> np.ndarray:
    """Center-crop 1280x720 -> 960x720 (4:3), then resize to 640x480."""
    h, w = img.shape[:2]
    assert (h, w) == (720, 1280), f"unexpected shape {img.shape}"
    x0 = (w - 960) // 2
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.resize(img[:, x0:x0 + 960], (640, 480), interpolation=interp)


def clip_invalid_depth(depth: np.ndarray, max_depth_m: float = MAX_VALID_DEPTH_M) -> np.ndarray:
    """Zero out physically-impossible depth readings (sensor noise/reflections)."""
    out = depth.copy()
    out[(out > max_depth_m) | ~np.isfinite(out)] = 0.0
    return out


class RealDepthDataset(Dataset):
    """One sample = one (rgb, ir, gt) triple. Same tensor shapes/dtypes as
    SyntheticDepthDataset: rgb (3,H,W), ir (1,H,W), gt (1,H,W), all float32."""
    # when three elements have the timestamp all together -- then it is consider as valid
    def __init__(self, root: str, augment: bool = False, base_seed: int = 0):
        self.root = Path(root)
        self.augment = augment
        self.base_seed = base_seed

        self.ffs_dir = self.root / "ffs_depth"
        self.intel_dir = self.root / "intel_depth"
        self.rgb_dir = self.root / "rgb"

        ffs_ts = {int(f.stem) for f in self.ffs_dir.glob("*.npy")}
        intel_ts = {int(f.stem) for f in self.intel_dir.glob("*.npy")}
        rgb_ts = {int(f.stem[5:]) for f in self.rgb_dir.glob("left_*.png")}

        self.timestamps = sorted(ffs_ts & intel_ts & rgb_ts)
        missing = (ffs_ts | intel_ts | rgb_ts) - set(self.timestamps)
        if missing:
            print(f"[RealDepthDataset] {len(missing)} timestamps missing from one folder, skipped")
        if not self.timestamps:
            raise RuntimeError(f"no aligned triples found under {root}")

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int):
        ts = self.timestamps[idx]

        rgb = np.array(Image.open(self.rgb_dir / f"left_{ts}.png"))
        intel = clip_invalid_depth(np.load(self.intel_dir / f"{ts}.npy"))
        gt = np.load(self.ffs_dir / f"{ts}.npy").astype(np.float32)  # already 640x480

        rgb = crop_720p_to_4_3(rgb, is_depth=False).astype(np.float32) / 255.0
        intel = crop_720p_to_4_3(intel, is_depth=True).astype(np.float32)

        if self.augment:
            from depth_augmentations import geometric_augmentation
            rng = np.random.default_rng(self.base_seed + idx)
            rgb, intel, gt = geometric_augmentation(rgb, intel, gt, rng)

        return (
            torch.from_numpy(rgb).permute(2, 0, 1).float(),
            torch.from_numpy(intel).unsqueeze(0).float(),
            torch.from_numpy(gt).unsqueeze(0).float(),
        )


if __name__ == "__main__":
    import sys

    ds = RealDepthDataset(sys.argv[1] if len(sys.argv) > 1 else "/content/data/gt_fixed")
    print(f"dataset length: {len(ds)}")

    rgb, ir, gt = ds[0]
    print(f"rgb: {tuple(rgb.shape)} [{rgb.min():.3f},{rgb.max():.3f}]")
    print(f"ir:  {tuple(ir.shape)} [{ir.min():.3f},{ir.max():.3f}] zeros={int((ir == 0).sum())}")
    print(f"gt:  {tuple(gt.shape)} [{gt.min():.3f},{gt.max():.3f}] zeros={int((gt == 0).sum())}")

    assert ir.max() <= MAX_VALID_DEPTH_M
    print("\nreal_dataset.py smoke test PASSED")
