"""
eval_baselines.py - two baselines to contextualize eval_metrics.py's numbers
on the held-out test group. Without these, "6.8cm MAE" has no way to be
judged as good or bad -- these answer "compared to what?"

Baseline A: raw IR depth vs ground truth, NO model at all.
    This is "what if we just used the sensor's own reading, uncorrected."
    If the trained model doesn't beat this, it isn't adding value.

Baseline B: Phase 1 model (trained only on synthetic data, never saw any
    real data) vs ground truth on this same held-out test group.
    This is "what if we skipped Phase 2 real-data fine-tuning entirely."
    If Phase 2's numbers aren't meaningfully better than this, the
    fine-tuning didn't help much.

Comparing eval_metrics.py's Phase-2 numbers against both of these tells
you whether the model (a) is doing anything useful at all, and (b) whether
the real-data fine-tuning specifically was worth doing.

Usage:
    python eval_baselines.py /content/data/new_extracted_unzipped
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from depth_correction_model import DepthCorrectionNet
from real_dataset import RealDepthDataset
from train_phase2 import N_TEST_GROUPS, split_test_train_val
from eval_metrics import compute_metrics, print_metrics

PHASE1_CKPT = Path("checkpoints_phase1/best.pt")
EPS = 1e-6


@torch.no_grad()
def compute_metrics_raw_ir(loader, device):
    """Same metric math as eval_metrics.compute_metrics, but 'pred' is just
    the raw IR depth itself -- no model involved."""
    total_abs_err = total_sq_err = total_rel_err = 0.0
    total_delta1 = total_delta2 = total_delta3 = 0.0
    total_valid_px = 0

    for rgb, ir, gt in loader:
        ir, gt = ir.to(device), gt.to(device)
        pred = ir  # the "model" here is just the sensor's own uncorrected reading

        valid = gt > 0
        pred_v, gt_v = pred[valid], gt[valid]
        n = pred_v.numel()
        if n == 0:
            continue

        abs_err = (pred_v - gt_v).abs()
        total_abs_err += abs_err.sum().item()
        total_sq_err += (abs_err ** 2).sum().item()
        total_rel_err += (abs_err / (gt_v + EPS)).sum().item()

        ratio = torch.maximum(pred_v / (gt_v + EPS), gt_v / (pred_v.abs() + EPS))
        total_delta1 += (ratio < 1.25).sum().item()
        total_delta2 += (ratio < 1.25 ** 2).sum().item()
        total_delta3 += (ratio < 1.25 ** 3).sum().item()

        total_valid_px += n

    if total_valid_px == 0:
        return None
    return {
        "mae_m": total_abs_err / total_valid_px,
        "rmse_m": (total_sq_err / total_valid_px) ** 0.5,
        "abs_rel": total_rel_err / total_valid_px,
        "delta1": total_delta1 / total_valid_px,
        "delta2": total_delta2 / total_valid_px,
        "delta3": total_delta3 / total_valid_px,
    }


def main():
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/data/new_extracted_unzipped")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    test_objects, _, _ = split_test_train_val(data_root, N_TEST_GROUPS)
    print(f"held-out test recordings: {test_objects}\n")

    test_datasets = [RealDepthDataset(data_root / name, augment=False) for name in test_objects]
    combined_loader = DataLoader(ConcatDataset(test_datasets), batch_size=4, shuffle=False)

    # --- Baseline A: raw IR, no model ---
    print("=== Baseline A: raw IR depth, no correction at all ===")
    m_a = compute_metrics_raw_ir(combined_loader, device)
    print_metrics("raw IR (uncorrected)", m_a)

    # --- Baseline B: Phase 1 model (synthetic-only, never saw real data) ---
    print("\n=== Baseline B: Phase 1 model (trained only on synthetic data) ===")
    model1 = DepthCorrectionNet().to(device)
    ckpt1 = torch.load(PHASE1_CKPT, map_location=device)
    model1.load_state_dict(ckpt1["model_state"])
    model1.eval()
    m_b = compute_metrics(model1, combined_loader, device)
    print_metrics("Phase 1 model (no real-data fine-tuning)", m_b)

    print("\n=== For comparison: run eval_metrics.py to get Phase 2's numbers ===")
    print("python eval_metrics.py " + str(data_root))


if __name__ == "__main__":
    main()
