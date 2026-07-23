"""
train_phase2.py - fine-tune Phase 1's model on real data, at native
720x1280 resolution, across 15 objects.

3 objects are held out ENTIRELY as test (never trained on) -- real unseen
scenes. The other 12 are pooled for train/val (80/20 split, done
separately per object since adjacent frames within an object are near-
duplicates and must not leak across the split).

Native res = ~9x the attention memory of the earlier 480x640 batch
(CrossAttentionFusion cost scales with pixels^2). BATCH_SIZE=1 as a result.
If it still OOMs: shrink EXPECTED_SHAPE in real_dataset.py (and add a
resize there), don't rewrite this file's logic.

Usage:
    python train_phase2.py /content/data/new_extracted_unzipped
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from depth_correction_model import DepthCorrectionNet
from losses import DepthCorrectionLoss
from real_dataset import RealDepthDataset

PHASE1_CKPT = Path("checkpoints_phase1/best.pt")
FREEZE_RGB_ENCODER = True  # set False to test whether unfreezing fixes extreme-lighting failures
CKPT_DIR = Path("checkpoints_phase2" if FREEZE_RGB_ENCODER else "checkpoints_phase2_unfrozen")
LOG_PATH = Path("phase2_loss_log.csv")

# which objects go where -- change these two lists if you want a different split
TEST_OBJECTS = ["05_foamandshirt", "07_SprayandWheel", "13_FoamRobotSpray"]
TRAIN_VAL_OBJECTS = [
    "01_clampjumpCoordsmeasureTapeRedRubberoutle", "02_Maskclampkeyboardspray",
    "03_RobotTapeHolderCanSprayWoodBlockRobotPar", "04_robotclampredrubberthingwaterbottlehydro",
    "06_HydroflaskTubeClampShirt", "08_Woodclampandcamerastand",
    "09_TapeHolderMeasureTapeKeyboardChargingSta", "10_JumpClampscamerastandwoodblockwrench",
    "11_ChargingStationCanJumperClampsWrench", "12_MaskKeyboardWoodBlock",
    "14_LightingLeftsidemaxbrightnorightsideligh", "15_LightingLeftsidemaxbrightnorightsideligh",
]

TRAIN_FRAC = 0.80
BATCH_SIZE, EPOCHS, LR = 1, 30, 1e-4  # LR lower than phase1: fine-tuning, not training from scratch
LOSS_WEIGHTS = dict(alpha=1.0, beta=0.5, gamma=0.5)
COMPONENTS = ("pixel", "grad", "ssim", "total")


def sequential_split(n, train_frac):
    """First train_frac of frames -> train, rest -> val. Not shuffled --
    adjacent frames within an object are near-identical."""
    n_train = int(n * train_frac)
    return list(range(n_train)), list(range(n_train, n))


def run_epoch(model, loader, criterion, optimizer, device, train, scaler=None):
    """One pass over `loader`. train=True updates weights; train=False just scores."""
    model.train(mode=train)
    totals = {k: 0.0 for k in COMPONENTS}
    for rgb, ir, gt in loader:
        rgb, ir, gt = rgb.to(device), ir.to(device), gt.to(device)
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss, parts = criterion(model(rgb, ir), gt)
        if train:
            optimizer.zero_grad()
            if scaler is not None:  # mixed precision path
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        for k in COMPONENTS:
            totals[k] += parts[k]
    return {k: v / len(loader) for k, v in totals.items()}


def build_train_val(data_root: Path):
    """Split each train/val object sequentially on its own, then glue all
    the train pieces together and all the val pieces together."""
    train_parts, val_parts = [], []
    for name in TRAIN_VAL_OBJECTS:
        obj_root = data_root / name
        train_ds = RealDepthDataset(obj_root, augment=True)   # augmented copy for training
        eval_ds = RealDepthDataset(obj_root, augment=False)    # clean copy for val
        train_idx, val_idx = sequential_split(len(train_ds), TRAIN_FRAC)
        train_parts.append(Subset(train_ds, train_idx))
        val_parts.append(Subset(eval_ds, val_idx))
        print(f"  {name}: train={len(train_idx)} val={len(val_idx)}")
    return ConcatDataset(train_parts), ConcatDataset(val_parts)


def main():
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/data/new_extracted_unzipped")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- data ---
    print("building train/val (pooled across objects, sequential split per object):")
    train_ds, val_ds = build_train_val(data_root)
    print(f"total: train={len(train_ds)} val={len(val_ds)}")

    test_datasets = {name: RealDepthDataset(data_root / name, augment=False) for name in TEST_OBJECTS}
    for name, ds in test_datasets.items():
        print(f"held-out test object '{name}': {len(ds)} frames (never trained on)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)   # shuffled: mix scenes each batch
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loaders = {name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for name, ds in test_datasets.items()}

    # --- model: start from phase 1, freeze rgb encoder ---
    model = DepthCorrectionNet().to(device)
    ckpt = torch.load(PHASE1_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded {PHASE1_CKPT} (phase1 epoch {ckpt['epoch']}, val_total={ckpt['val_total']:.4f})")

    if FREEZE_RGB_ENCODER:
        for p in model.rgb_encoder.parameters():
            p.requires_grad = False  # real RGB is close enough to synthetic RGB to reuse as-is -- but see extreme-lighting failure notes above
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"FREEZE_RGB_ENCODER={FREEZE_RGB_ENCODER}")
    label = "frozen" if FREEZE_RGB_ENCODER else "unfrozen (trainable)"
    print(f"{label} (rgb_encoder): {sum(p.numel() for p in model.rgb_encoder.parameters()):,} params | "
          f"trainable: {sum(p.numel() for p in trainable):,} params")

    # --- training setup ---
    criterion = DepthCorrectionLoss(**LOSS_WEIGHTS).to(device)
    optimizer = torch.optim.Adam(trainable, lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))  # mixed precision -> less memory
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"\n--- Phase 2 fine-tuning: {EPOCHS} epochs --- weights: {LOSS_WEIGHTS}\n")

    # --- training loop ---
    history, best_val = [], float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_parts = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_parts = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()  # release fragmented memory between epochs

        history.append({"epoch": epoch,
                         **{f"train_{k}": v for k, v in train_parts.items()},
                         **{f"val_{k}": v for k, v in val_parts.items()}})

        fmt = lambda p: " ".join(f"{k}={p[k]:.4f}" for k in COMPONENTS)
        print(f"epoch {epoch:3d}/{EPOCHS} ({time.time() - t0:5.1f}s)  "
              f"train: {fmt(train_parts)}  |  val: {fmt(val_parts)}")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "val_total": val_parts["total"]},
                   CKPT_DIR / "last.pt")               # always overwritten: latest weights
        if val_parts["total"] < best_val:
            best_val = val_parts["total"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_total": best_val},
                       CKPT_DIR / "best.pt")            # only overwritten on a new record

    # --- save the loss curve for later plotting ---
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"\nloss history written to {LOG_PATH.resolve()}")
    print(f"best val_total: {best_val:.4f}")

    # --- final report card: best.pt on each held-out object, scored separately ---
    best_ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print("\n--- final test evaluation on held-out objects (best.pt) ---")
    for name, loader in test_loaders.items():
        test_parts = run_epoch(model, loader, criterion, optimizer, device, train=False)
        print(f"  {name}: " + " ".join(f"{k}={test_parts[k]:.4f}" for k in COMPONENTS))

    print(f"\ncheckpoints saved to {CKPT_DIR.resolve()}")


if __name__ == "__main__":
    main()