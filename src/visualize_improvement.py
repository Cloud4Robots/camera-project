"""
visualize_improvement.py - two things this adds on top of eval_metrics.py:

1. Threshold-based accuracy: "what % of pixels are within X cm of the true
   depth" for a few concrete thresholds (1cm/2cm/5cm/10cm), instead of the
   relative delta1/delta2/delta3 ratios. More intuitive for a quick read.

2. Visual before/after comparison: for a few held-out test frames, plots
   RGB / raw IR / Phase 1 prediction / Phase 2 prediction / ground truth
   side by side, plus error heatmaps for raw-IR-vs-gt and phase2-vs-gt so
   you can SEE where the correction helped, not just read a number.

Usage:
    python visualize_improvement.py /path/to/new_extracted_unzipped
    python visualize_improvement.py /path/to/new_extracted_unzipped --n_frames 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from depth_correction_model import DepthCorrectionNet
from real_dataset import RealDepthDataset
from train_phase2 import N_TEST_GROUPS, split_test_train_val

PHASE1_CKPT = Path("checkpoints_phase1/best.pt")
PHASE2_CKPT = Path("checkpoints_phase2/best.pt")

# absolute-error thresholds to report, in meters
THRESHOLDS_M = [0.01, 0.02, 0.05, 0.10]


def threshold_accuracy(pred, gt, thresholds_m):
    """% of valid pixels where |pred - gt| < threshold, for each threshold."""
    valid = gt > 0
    abs_err = (pred[valid] - gt[valid]).abs()
    n = abs_err.numel()
    return {f"within_{int(t*100)}cm": (abs_err < t).float().mean().item() * 100 for t in thresholds_m}, n


def print_threshold_table(name, pred, gt):
    accs, n = threshold_accuracy(pred, gt, THRESHOLDS_M)
    print(f"  {name} ({n} valid pixels):")
    for k, v in accs.items():
        print(f"    {k:12s}: {v:5.1f}% of pixels")


@torch.no_grad()
def load_models(device):
    model1 = DepthCorrectionNet().to(device)
    model1.load_state_dict(torch.load(PHASE1_CKPT, map_location=device)["model_state"])
    model1.eval()

    model2 = DepthCorrectionNet().to(device)
    model2.load_state_dict(torch.load(PHASE2_CKPT, map_location=device)["model_state"])
    model2.eval()

    return model1, model2


def visualize_frame(rgb, ir, gt, pred1, pred2, idx, out_dir: Path):
    """rgb: (3,H,W) tensor 0-1; ir/gt/pred1/pred2: (1,H,W) tensor, meters."""
    rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
    ir_np = ir.squeeze(0).cpu().numpy()
    gt_np = gt.squeeze(0).cpu().numpy()
    p1_np = pred1.squeeze(0).cpu().numpy()
    p2_np = pred2.squeeze(0).cpu().numpy()

    vmax = np.percentile(gt_np[gt_np > 0], 95) if (gt_np > 0).any() else 2.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title("RGB")

    axes[0, 1].imshow(ir_np, cmap="viridis", vmin=0, vmax=vmax)
    axes[0, 1].set_title("Raw IR (uncorrected input)")

    axes[0, 2].imshow(gt_np, cmap="viridis", vmin=0, vmax=vmax)
    axes[0, 2].set_title("Ground truth")

    axes[1, 0].imshow(p1_np, cmap="viridis", vmin=0, vmax=vmax)
    axes[1, 0].set_title("Phase 1 prediction (synthetic-only)")

    axes[1, 1].imshow(p2_np, cmap="viridis", vmin=0, vmax=vmax)
    axes[1, 1].set_title("Phase 2 prediction (after real-data fine-tuning)")

    # error heatmap: how much better is phase2 than raw IR, pixel by pixel
    valid = gt_np > 0
    err_raw = np.abs(ir_np - gt_np)
    err_p2 = np.abs(p2_np - gt_np)
    improvement = np.zeros_like(gt_np)
    improvement[valid] = err_raw[valid] - err_p2[valid]  # positive = phase2 better

    im = axes[1, 2].imshow(improvement, cmap="RdBu", vmin=-0.2, vmax=0.2)
    axes[1, 2].set_title("Improvement (blue=better after training, red=worse)")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()

    out_path = out_dir / f"comparison_frame_{idx}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("--n_frames", type=int, default=3, help="how many held-out frames to visualize")
    parser.add_argument("--out_dir", default="viz_output")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    test_objects, _, _ = split_test_train_val(data_root, N_TEST_GROUPS)
    print(f"held-out test recordings: {test_objects}")

    model1, model2 = load_models(device)

    # gather threshold accuracy across the whole held-out set, and pick a
    # few frames (evenly spaced) to actually visualize
    all_rgb, all_ir, all_gt, all_p1, all_p2 = [], [], [], [], []

    for name in test_objects:
        ds = RealDepthDataset(data_root / name, augment=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        for rgb, ir, gt in loader:
            rgb, ir, gt = rgb.to(device), ir.to(device), gt.to(device)
            with torch.no_grad():
                p1 = model1(rgb, ir)
                p2 = model2(rgb, ir)
            all_rgb.append(rgb[0].cpu())
            all_ir.append(ir[0].cpu())
            all_gt.append(gt[0].cpu())
            all_p1.append(p1[0].cpu())
            all_p2.append(p2[0].cpu())

    ir_cat = torch.stack(all_ir)
    gt_cat = torch.stack(all_gt)
    p1_cat = torch.stack(all_p1)
    p2_cat = torch.stack(all_p2)

    print(f"\n--- threshold-based accuracy (held-out test set, {len(all_gt)} frames) ---")
    print_threshold_table("raw IR (uncorrected)", ir_cat, gt_cat)
    print_threshold_table("Phase 1 (synthetic-only)", p1_cat, gt_cat)
    print_threshold_table("Phase 2 (real-data fine-tuned)", p2_cat, gt_cat)

    n = len(all_rgb)
    n_frames = min(args.n_frames, n)
    indices = np.linspace(0, n - 1, n_frames, dtype=int)

    print(f"\n--- saving {n_frames} side-by-side comparison images to {out_dir}/ ---")
    for idx in indices:
        visualize_frame(all_rgb[idx], all_ir[idx], all_gt[idx], all_p1[idx], all_p2[idx], idx, out_dir)


if __name__ == "__main__":
    main()
