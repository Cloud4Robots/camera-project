"""
real_dataset.py - loads one (rgb, ir, gt) frame at a time for a single
object's capture folder. Native resolution this batch: 720x1280, no
crop/resize (if memory becomes an issue, shrink EXPECTED_SHAPE and add a
resize here rather than rewriting this file).

Folder layout expected:
    root/rgb_rect/left_{ts}.webp
    root/intel_depth_rect/{ts}.npz   (key "depth")  <- noisy raw IR input
    root/ffs_depth/{ts}.npz          (key "depth")  <- ground truth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MAX_VALID_DEPTH_M = 3.0          # anything deeper than this is sensor noise, not real
EXPECTED_SHAPE = (720, 1280)     # native res this batch; lower + add resize if OOM


def clip_invalid_depth(depth: np.ndarray, max_depth_m: float = MAX_VALID_DEPTH_M) -> np.ndarray:
    """Zero out impossible depth readings (e.g. the 65m spikes seen in one object)."""
    out = depth.copy()
    out[(out > max_depth_m) | ~np.isfinite(out)] = 0.0
    return out


def _load_npz_depth(path: Path) -> np.ndarray:
    """npz stores the depth array under the key 'depth'."""
    return np.load(path)["depth"].astype(np.float32)


class RealDepthDataset(Dataset):
    """One object's folder = one dataset. __getitem__ returns one aligned
    (rgb, ir, gt) triple as tensors, ready for the model."""

    def __init__(self, root: str, augment: bool = False, base_seed: int = 0):
        self.root = Path(root)
        self.augment = augment      # turn on flip/rotate augmentation (train split only)
        self.base_seed = base_seed  # makes augmentation reproducible per-index

        self.ffs_dir = self.root / "ffs_depth"
        self.intel_dir = self.root / "intel_depth_rect"
        self.rgb_dir = self.root / "rgb_rect"

        # only keep frames that exist in ALL THREE folders (matched by timestamp)
        ffs_ts = {int(f.stem) for f in self.ffs_dir.glob("*.npz")}
        intel_ts = {int(f.stem) for f in self.intel_dir.glob("*.npz")}
        rgb_ts = {int(f.stem[5:]) for f in self.rgb_dir.glob("left_*.webp")}  # strip "left_" prefix

        self.timestamps = sorted(ffs_ts & intel_ts & rgb_ts)
        missing = (ffs_ts | intel_ts | rgb_ts) - set(self.timestamps)
        if missing:
            print(f"[RealDepthDataset] {self.root.name}: {len(missing)} timestamps missing from one folder, skipped")
        if not self.timestamps:
            raise RuntimeError(f"no aligned triples found under {root}")

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, idx: int):
        ts = self.timestamps[idx]

        # load the three files for this timestamp
        rgb = np.array(Image.open(self.rgb_dir / f"left_{ts}.webp").convert("RGB"))
        intel = clip_invalid_depth(_load_npz_depth(self.intel_dir / f"{ts}.npz"))
        gt = _load_npz_depth(self.ffs_dir / f"{ts}.npz")

        # fail loudly (not silently) if a frame doesn't match the expected shape
        assert rgb.shape[:2] == EXPECTED_SHAPE, f"{self.root.name}/{ts}: bad rgb shape {rgb.shape}"
        assert intel.shape == EXPECTED_SHAPE, f"{self.root.name}/{ts}: bad intel shape {intel.shape}"
        assert gt.shape == EXPECTED_SHAPE, f"{self.root.name}/{ts}: bad gt shape {gt.shape}"

        rgb = rgb.astype(np.float32) / 255.0  # normalize pixels to 0-1

        if self.augment:
            from depth_augmentations import geometric_augmentation
            rng = np.random.default_rng(self.base_seed + idx)
            rgb, intel, gt = geometric_augmentation(rgb, intel, gt, rng)

        # convert to the (C,H,W) tensor layout PyTorch expects
        return (
            torch.from_numpy(rgb).permute(2, 0, 1).float(),
            torch.from_numpy(intel).unsqueeze(0).float(),
            torch.from_numpy(gt).unsqueeze(0).float(),
        )


if __name__ == "__main__":
    # quick manual check: load one object, print shapes/ranges, confirm no crash
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "/content/data/new_extracted_unzipped/01_clampjumpCoordsmeasureTapeRedRubberoutle"
    ds = RealDepthDataset(root)
    print(f"dataset length: {len(ds)}")

    rgb, ir, gt = ds[0]
    print(f"rgb: {tuple(rgb.shape)} [{rgb.min():.3f},{rgb.max():.3f}]")
    print(f"ir:  {tuple(ir.shape)} [{ir.min():.3f},{ir.max():.3f}] zeros={int((ir == 0).sum())}")
    print(f"gt:  {tuple(gt.shape)} [{gt.min():.3f},{gt.max():.3f}] zeros={int((gt == 0).sum())}")

    assert ir.max() <= MAX_VALID_DEPTH_M
    print("\nreal_dataset.py smoke test PASSED")