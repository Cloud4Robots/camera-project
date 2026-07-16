"""
train_phase1.py — Phase 1 warmup: pretrain on synthetic data, track
pixel/grad/ssim convergence per epoch. Usage: python train_phase1.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from depth_correction_model import DepthCorrectionNet
from losses import DepthCorrectionLoss
from synthetic_dataset import SyntheticDepthDataset

IMG_SIZE = 256  # bumped from 128: at 128px, Rhys's 3-8mm rod calibration collapses to 1px either way
TRAIN_SIZE, VAL_SIZE = 400, 80
BATCH_SIZE, EPOCHS, LR = 8, 20, 1e-3
LOSS_WEIGHTS = dict(alpha=1.0, beta=0.5, gamma=0.5)  # from 7/14 sanity check
CKPT_DIR, LOG_PATH, SEED = Path("checkpoints_phase1"), Path("phase1_loss_log.csv"), 0

COMPONENTS = ("pixel", "grad", "ssim", "total")

#tell the model if it should be in training mode or not
def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict:
    model.train(mode=train)
    totals = {k: 0.0 for k in COMPONENTS}
# the loader will return a batch of data, which is a tuple of (rgb, ir, gt)
    for rgb, ir, gt in loader:
        rgb, ir, gt = rgb.to(device), ir.to(device), gt.to(device)
        if train:
            optimizer.zero_grad()
        # if it is the training mode -- refresh from the previous state of the model 
        # and compute the loss and the gradients
        with torch.set_grad_enabled(train):
            loss, parts = criterion(model(rgb, ir), gt)
        if train:
            loss.backward()
            optimizer.step()
        for k in COMPONENTS:
            totals[k] += parts[k]

    return {k: v / len(loader) for k, v in totals.items()}


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader = DataLoader(
        SyntheticDepthDataset(TRAIN_SIZE, IMG_SIZE, IMG_SIZE, base_seed=SEED),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        SyntheticDepthDataset(VAL_SIZE, IMG_SIZE, IMG_SIZE, base_seed=SEED + 100_000),
        batch_size=BATCH_SIZE, shuffle=False)
    # build up the model training dataset and validation dataset with the synthetic data generator
    model = DepthCorrectionNet().to(device)
    criterion = DepthCorrectionLoss(**LOSS_WEIGHTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    CKPT_DIR.mkdir(exist_ok=True)

    print(f"\n--- Phase 1 warmup: {EPOCHS} epochs, {TRAIN_SIZE} train / {VAL_SIZE} val --- weights: {LOSS_WEIGHTS}\n")

    history, best_val = [], float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_parts = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_parts = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
    # train through the epoches then validate the model and 
    # save the best model based on the validation loss
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

    print("\n--- convergence check (train, epoch 1 vs last) ---")
    first, last, all_ok = history[0], history[-1], True
    for k in ("pixel", "grad", "ssim"):
        f0, f1 = first[f"train_{k}"], last[f"train_{k}"]
        ok = f1 < f0
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:6s} {f0:.4f} -> {f1:.4f} ({(f0 - f1) / f0 * 100:+.1f}%)")
        all_ok &= ok

    print("\n" + ("ALL THREE LOSS COMPONENTS DECREASING - architecture/loss look sound" if all_ok
                   else "AT LEAST ONE COMPONENT DID NOT DECREASE - check it before Phase 2"))
    print(f"checkpoints saved to {CKPT_DIR.resolve()}")
# take the 1st round three losses comparing with the last round three losses to see if the model is converging or not

if __name__ == "__main__":
    main()