"""
eval_metrics.py - human-readable accuracy metrics for the held-out test
group(s), on top of the raw loss numbers train_phase2.py already prints.

"Accurate" is defined by ACCURACY_THRESHOLD_M: a pixel counts as accurate
if |pred - gt| < ACCURACY_THRESHOLD_M. Change this one constant to match
whatever tolerance actually matters for the downstream task (e.g. how
precise the robot arm's grasping needs to be) -- there's no universal
"correct" threshold, it depends on what the depth is used for.

Also reports the standard depth-estimation paper metrics for reference:
  MAE, RMSE, AbsRel, and the relative delta1/delta2/delta3 (ratio-based,
  as opposed to the absolute-threshold "accuracy" above).

Only computed on valid pixels (gt > 0) -- the zeroed-out invalid regions
(stereo-matching blind spots, clipped sensor outliers) are excluded so
they don't distort the numbers.

Usage:
    python eval_metrics.py /content/data/new_extracted_unzipped
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from depth_correction_model import DepthCorrectionNet
from real_dataset import RealDepthDataset
from train_phase2 import N_TEST_GROUPS, split_test_train_val

CKPT_PATH = Path("checkpoints_phase2/best.pt")
EPS = 1e-6

# --- the single number that answers "is this pixel accurate?" ---
# change this to match your actual tolerance requirement
ACCURACY_THRESHOLD_M = 0.05  # 5cm


@torch.no_grad()
def compute_metrics(model, loader, device, accuracy_threshold_m=ACCURACY_THRESHOLD_M):
    """Accumulate metrics over every batch, weighted by valid pixel count
    (not by batch count) so images with more/fewer valid pixels are
    weighted fairly."""
    total_abs_err = total_sq_err = total_rel_err = 0.0
    total_accurate = 0.0
    total_delta1 = total_delta2 = total_delta3 = 0.0
    total_valid_px = 0

    for rgb, ir, gt in loader:
        rgb, ir, gt = rgb.to(device), ir.to(device), gt.to(device)
        pred = model(rgb, ir)

        valid = gt > 0
        pred_v, gt_v = pred[valid], gt[valid]
        n = pred_v.numel()
        if n == 0:
            continue

        abs_err = (pred_v - gt_v).abs()
        total_abs_err += abs_err.sum().item()
        total_sq_err += (abs_err ** 2).sum().item()
        total_rel_err += (abs_err / (gt_v + EPS)).sum().item()
        total_accurate += (abs_err < accuracy_threshold_m).sum().item()

        ratio = torch.maximum(pred_v / (gt_v + EPS), gt_v / (pred_v.abs() + EPS))
        total_delta1 += (ratio < 1.25).sum().item()
        total_delta2 += (ratio < 1.25 ** 2).sum().item()
        total_delta3 += (ratio < 1.25 ** 3).sum().item()

        total_valid_px += n

    if total_valid_px == 0:
        return None

    return {
        "accuracy": total_accurate / total_valid_px,  # THE headline number
        "mae_m": total_abs_err / total_valid_px,
        "rmse_m": (total_sq_err / total_valid_px) ** 0.5,
        "abs_rel": total_rel_err / total_valid_px,
        "delta1": total_delta1 / total_valid_px,
        "delta2": total_delta2 / total_valid_px,
        "delta3": total_delta3 / total_valid_px,
        "valid_pixels": total_valid_px,
    }


def print_metrics(name, m, accuracy_threshold_m=ACCURACY_THRESHOLD_M):
    if m is None:
        print(f"  {name}: no valid pixels found, skipping")
        return
    print(f"  {name}:")
    print(f"    accuracy (err < {int(accuracy_threshold_m*100)}cm): {m['accuracy']*100:.1f}%  <-- headline number")
    print(f"    MAE     : {m['mae_m']*100:.1f} cm")
    print(f"    RMSE    : {m['rmse_m']*100:.1f} cm")
    print(f"    AbsRel  : {m['abs_rel']*100:.1f}%")
    print(f"    delta1  : {m['delta1']*100:.1f}%  (relative version: within 1.25x of true depth)")
    print(f"    delta2  : {m['delta2']*100:.1f}%")
    print(f"    delta3  : {m['delta3']*100:.1f}%")


def main():
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/data/new_extracted_unzipped")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"accuracy threshold: {ACCURACY_THRESHOLD_M*100:.0f}cm (change ACCURACY_THRESHOLD_M to adjust)\n")

    test_objects, _, _ = split_test_train_val(data_root, N_TEST_GROUPS)
    print(f"held-out test recordings: {test_objects}\n")

    model = DepthCorrectionNet().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {CKPT_PATH} (phase2 epoch {ckpt['epoch']}, val_total={ckpt['val_total']:.4f})\n")

    print("--- per-recording metrics (held-out test group) ---")
    for name in test_objects:
        ds = RealDepthDataset(data_root / name, augment=False)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        m = compute_metrics(model, loader, device)
        print_metrics(name, m)

    print("\n--- combined across all held-out test recordings ---")
    all_datasets = [RealDepthDataset(data_root / name, augment=False) for name in test_objects]
    combined_loader = DataLoader(torch.utils.data.ConcatDataset(all_datasets), batch_size=4, shuffle=False)
    m = compute_metrics(model, combined_loader, device)
    print_metrics("ALL", m)


if __name__ == "__main__":
    main()
