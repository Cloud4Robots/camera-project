"""
eval_diagnostics.py - drills into WHERE the model's errors are concentrated:
by depth range (is far/background worse than near/foreground?) and by
individual frame (are a few frames much worse than the rest, or is it
spread evenly?).

Run this after eval_metrics.py shows a concerning aggregate number, or
after spotting a bad-looking frame in visualize_improvement.py, to find
out whether it's a systematic pattern or a handful of outlier frames.

Usage:
    python eval_diagnostics.py /path/to/new_extracted_unzipped
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

# depth bins to break accuracy down by, in meters -- adjust to match your workspace
DEPTH_BINS = [(0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 100.0)]
ACCURACY_THRESHOLD_M = 0.05


@torch.no_grad()
def accuracy_by_depth_bin(model, loader, device):
    """accuracy + pixel count, split by which depth bin the GROUND TRUTH
    value falls into. Compares accuracy near vs far."""
    bin_correct = {b: 0 for b in DEPTH_BINS}
    bin_total = {b: 0 for b in DEPTH_BINS}

    for rgb, ir, gt in loader:
        rgb, ir, gt = rgb.to(device), ir.to(device), gt.to(device)
        pred = model(rgb, ir)

        valid = gt > 0
        pred_v, gt_v = pred[valid], gt[valid]
        abs_err = (pred_v - gt_v).abs()
        correct = abs_err < ACCURACY_THRESHOLD_M

        for lo, hi in DEPTH_BINS:
            in_bin = (gt_v >= lo) & (gt_v < hi)
            bin_total[(lo, hi)] += in_bin.sum().item()
            bin_correct[(lo, hi)] += (correct & in_bin).sum().item()

    print(f"\n--- accuracy by ground-truth depth range (threshold={ACCURACY_THRESHOLD_M*100:.0f}cm) ---")
    for lo, hi in DEPTH_BINS:
        n = bin_total[(lo, hi)]
        if n == 0:
            print(f"  {lo:.1f}-{hi:.1f}m: no pixels in this range")
            continue
        acc = bin_correct[(lo, hi)] / n
        label = f"{lo:.1f}-{hi:.1f}m" if hi < 100 else f">{lo:.1f}m"
        print(f"  {label:10s}: {acc*100:5.1f}% accurate  ({n:,} pixels)")


@torch.no_grad()
def per_frame_accuracy(model, ds, device, name):
    """accuracy for EACH individual frame, to spot outlier frames rather
    than an averaged-out aggregate."""
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    results = []
    for i, (rgb, ir, gt) in enumerate(loader):
        rgb, ir, gt = rgb.to(device), ir.to(device), gt.to(device)
        pred = model(rgb, ir)
        valid = gt > 0
        if valid.sum() == 0:
            continue
        abs_err = (pred[valid] - gt[valid]).abs()
        acc = (abs_err < ACCURACY_THRESHOLD_M).float().mean().item()
        results.append((i, acc))

    results.sort(key=lambda x: x[1])  # worst first
    print(f"\n--- per-frame accuracy for '{name}' ({len(results)} frames, worst 10 shown) ---")
    for idx, acc in results[:10]:
        print(f"  frame {idx:4d}: {acc*100:5.1f}% accurate")
    accs = [a for _, a in results]
    print(f"\n  mean: {sum(accs)/len(accs)*100:.1f}%  min: {min(accs)*100:.1f}%  max: {max(accs)*100:.1f}%")
    n_bad = sum(1 for a in accs if a < 0.5)
    print(f"  frames with <50% accuracy: {n_bad} / {len(accs)} ({n_bad/len(accs)*100:.1f}%)")


def main():
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/data/new_extracted_unzipped")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    test_objects, _, _ = split_test_train_val(data_root, N_TEST_GROUPS)
    print(f"held-out test recordings: {test_objects}")

    model = DepthCorrectionNet().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_datasets = [RealDepthDataset(data_root / name, augment=False) for name in test_objects]
    combined_loader = DataLoader(torch.utils.data.ConcatDataset(all_datasets), batch_size=4, shuffle=False)
    accuracy_by_depth_bin(model, combined_loader, device)

    for name in test_objects:
        ds = RealDepthDataset(data_root / name, augment=False)
        per_frame_accuracy(model, ds, device, name)


if __name__ == "__main__":
    main()
