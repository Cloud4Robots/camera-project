"""
train_phase2.py — Phase 2: fine-tune on real data. Loads Phase 1's best.pt,
freezes the RGB encoder (real RGB is close enough to synthetic RGB to
transfer), and fine-tunes IR encoder + fusion + decoder on the real
715-frame dataset (real IR noise differs from Phase 1's synthetic noise).

Split is sequential (not shuffled) — frames come from one continuous
recording, so a random split would leak near-duplicate frames across
train/val. train 80% / val 20%. No test split yet (no independent
real-world capture exists); sequential_split supports a 3-way split
once one does.

Usage:
    python train_phase2.py /content/data/gt_fixed
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from depth_correction_model import DepthCorrectionNet
from losses import DepthCorrectionLoss
from real_dataset import RealDepthDataset

PHASE1_CKPT = Path("checkpoints_phase1/best.pt")
CKPT_DIR = Path("checkpoints_phase2")
LOG_PATH = Path("phase2_loss_log.csv")

TRAIN_FRAC = 0.80
BATCH_SIZE, EPOCHS, LR = 8, 30, 1e-4  # lower LR than phase1: fine-tuning, not training from scratch
LOSS_WEIGHTS = dict(alpha=1.0, beta=0.5, gamma=0.5)
COMPONENTS = ("pixel", "grad", "ssim", "total")

# split the data into train and validation set. the ratio is 8 : 2. Cut the order by the range. 

def sequential_split(n, train_frac, val_frac=None):
    n_train = int(n * train_frac)
    train_idx = list(range(n_train))
    if val_frac is None:
        return train_idx, list(range(n_train, n))
    n_val = int(n * val_frac)
    return train_idx, list(range(n_train, n_train + n_val)), list(range(n_train + n_val, n))

# this is similar to phase 1, run through the one complete loader, and return the average loss for each epoch. 
def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(mode=train)
    totals = {k: 0.0 for k in COMPONENTS}
    for rgb, ir, gt in loader:
        rgb, ir, gt = rgb.to(device), ir.to(device), gt.to(device)
        with torch.set_grad_enabled(train):
            loss, parts = criterion(model(rgb, ir), gt)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #this is where the model is updated based on the gradients computed from the loss
        for k in COMPONENTS:
            totals[k] += parts[k]
    return {k: v / len(loader) for k, v in totals.items()}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/content/data/gt_fixed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # two dataset instances (deterministic ordering): augment for train, not for val
    train_ds, eval_ds = RealDepthDataset(root, augment=True), RealDepthDataset(root, augment=False)
    train_idx, val_idx = sequential_split(len(train_ds), TRAIN_FRAC)
    print(f"split: train={len(train_idx)} val={len(val_idx)} (sequential, no held-out test yet)")

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(eval_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = DepthCorrectionNet().to(device)
    ckpt = torch.load(PHASE1_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded {PHASE1_CKPT} (phase1 epoch {ckpt['epoch']}, val_total={ckpt['val_total']:.4f})")

    #RGB does not have to be fine-tuned because the real RGB is close enough to synthetic RGB. 
    # So freeze the RGB encoder and only fine-tune the IR encoder, fusion, and decoder.
    for p in model.rgb_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"frozen (rgb_encoder): {sum(p.numel() for p in model.rgb_encoder.parameters()):,} params | "
          f"trainable: {sum(p.numel() for p in trainable):,} params")

    criterion = DepthCorrectionLoss(**LOSS_WEIGHTS).to(device)
    optimizer = torch.optim.Adam(trainable, lr=LR)
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"\n--- Phase 2 fine-tuning: {EPOCHS} epochs --- weights: {LOSS_WEIGHTS}\n")

    history, best_val = [], float("inf")
    # train changes the model parameters based on the training data, 
    # and then validate the model on the validation data (no changes in parameters).
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_parts = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_parts = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch,
                         **{f"train_{k}": v for k, v in train_parts.items()},
                         **{f"val_{k}": v for k, v in val_parts.items()}})

        fmt = lambda p: " ".join(f"{k}={p[k]:.4f}" for k in COMPONENTS)
        print(f"epoch {epoch:3d}/{EPOCHS} ({time.time() - t0:5.1f}s)  "
              f"train: {fmt(train_parts)}  |  val: {fmt(val_parts)}")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "val_total": val_parts["total"]},
                   CKPT_DIR / "last.pt")
        if val_parts["total"] < best_val:
            best_val = val_parts["total"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_total": best_val},
                       CKPT_DIR / "best.pt")

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"\nloss history written to {LOG_PATH.resolve()}")
    print(f"best val_total: {best_val:.4f}")
    print(f"checkpoints saved to {CKPT_DIR.resolve()}")


if __name__ == "__main__":
    main()