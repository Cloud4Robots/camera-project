"""
check_phase1_on_real.py — sanity check before writing the Phase 2 training
loop. Loads Phase 1's best.pt and runs a forward-only pass over the real
715-frame dataset to see the current loss gap (synthetic-trained model on
real data), and to catch any shape/dtype mismatches before training starts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from depth_correction_model import DepthCorrectionNet
from losses import DepthCorrectionLoss
from real_dataset import RealDepthDataset

CKPT_PATH = Path("checkpoints_phase1/best.pt")
COMPONENTS = ("pixel", "grad", "ssim", "total")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/content/data/gt_fixed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    print(f"loaded {CKPT_PATH} (epoch {ckpt['epoch']}, phase1 val_total={ckpt['val_total']:.4f})")

    model = DepthCorrectionNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = DepthCorrectionLoss().to(device)

    ds = RealDepthDataset(root, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    print(f"real dataset: {len(ds)} frames")

    totals = {k: 0.0 for k in COMPONENTS}
    with torch.no_grad():
        for rgb, ir, gt in loader:
            rgb, ir, gt = rgb.to(device), ir.to(device), gt.to(device)
            pred = model(rgb, ir)
            _, parts = criterion(pred, gt)
            for k in COMPONENTS:
                totals[k] += parts[k] * rgb.shape[0]

    avg = {k: v / len(ds) for k, v in totals.items()}
    print("\n--- Phase 1 model on real data (forward only, no training) ---")
    for k in COMPONENTS:
        print(f"  {k:6s}: {avg[k]:.4f}")

    print(f"\nfor reference, Phase 1's synthetic val_total was {ckpt['val_total']:.4f}")
    print("a real_total noticeably higher than that is the expected domain gap Phase 2 should close.")


if __name__ == "__main__":
    main()