"""
train_phase2.py - fine-tune Phase 1's model on real data, at native
720x1280 resolution, across auto-discovered objects, grouped by physical
scene so lighting variants of the same objects never split across
train/test.

Many recordings are the SAME physical objects re-shot under different
lighting (e.g. "..._charging_station_tube_mask" appears 6 times with
different lighting prefixes). Splitting those individually would leak the
same scene into both train and test -- test would measure "recognizes this
object under a new light," not "recognizes an unseen object." So objects
are grouped by their canonical (lighting-stripped) descriptor, and
N_TEST_GROUPS whole groups are held out entirely.

Native res = ~9x the attention memory of the earlier 480x640 batch
(CrossAttentionFusion cost scales with pixels^2). BATCH_SIZE=1 as a
result -- raise it if nvidia-smi shows memory headroom, since batch=1
underuses the GPU.

Usage:
    python train_phase2.py /content/data/new_extracted_unzipped
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from depth_correction_model import DepthCorrectionNet
from losses import DepthCorrectionLoss
from real_dataset import RealDepthDataset

PHASE1_CKPT = Path("checkpoints_phase1/best.pt")
CKPT_DIR = Path("checkpoints_phase2")
LOG_PATH = Path("phase2_loss_log.csv")

N_TEST_GROUPS = 1  # how many whole scene-groups (all lighting variants included) to hold out as test

# known lighting-variant phrases that get stripped to find the canonical
# (lighting-independent) scene descriptor -- add more here if new variants appear
LIGHTING_VARIANTS = [
    "lighting_left_side_max_bright_no_right_side_lighting_",
    "lighting_right_side_max_bright_no_left_side_lighting_",
    "lighting_no_lighting_both_sides_",
    "lighting_max_bright_lighting_both_sides_",
    "lighting_regular_lighting_",
    "lighting_overhead_lighting_",
]

TRAIN_FRAC = 0.80
BATCH_SIZE, EPOCHS, LR = 1, 20, 1e-4
LOSS_WEIGHTS = dict(alpha=1.0, beta=0.5, gamma=0.5)
COMPONENTS = ("pixel", "grad", "ssim", "total")


def discover_objects(root: Path) -> list[str]:
    """Any folder with a ffs_depth subfolder, at any nesting depth."""
    ffs_dirs = list(root.rglob("ffs_depth"))
    return sorted({str(f.parent.relative_to(root)) for f in ffs_dirs})


def canonical_group_key(folder_name: str) -> str:
    """Strip the leading number and any known lighting-variant phrase,
    leaving just the physical-object descriptor. Recordings with no
    lighting variant (the original object-only batch) keep their full
    descriptor and end up as their own single-member group."""
    name = re.sub(r"^\d+_", "", folder_name)
    for variant in LIGHTING_VARIANTS:
        if name.startswith(variant):
            return name[len(variant):]
    return name


def split_test_train_val(data_root: Path, n_test_groups: int):
    """Group objects by canonical_group_key, then hold out the n_test_groups
    smallest-by-total-frame-count groups (ALL their lighting variants)
    entirely as test."""
    all_objects = discover_objects(data_root)

    def frame_count(name):
        return len(list((data_root / name / "ffs_depth").glob("*.npz")))

    groups: dict[str, list[str]] = {}
    for name in all_objects:
        groups.setdefault(canonical_group_key(name), []).append(name)

    if len(groups) <= n_test_groups:
        raise RuntimeError(f"only {len(groups)} scene groups found, need more than {n_test_groups} to hold any out")

    ranked = sorted(groups.items(), key=lambda kv: sum(frame_count(n) for n in kv[1]))
    test_objects = [n for _, members in ranked[:n_test_groups] for n in members]
    train_val_objects = [n for _, members in ranked[n_test_groups:] for n in members]
    return test_objects, train_val_objects, groups


def sequential_split(n, train_frac):
    n_train = int(n * train_frac)
    return list(range(n_train)), list(range(n_train, n))


def run_epoch(model, loader, criterion, optimizer, device, train, scaler=None):
    model.train(mode=train)
    totals = {k: 0.0 for k in COMPONENTS}
    for rgb, ir, gt in tqdm(loader, desc="train" if train else "val", leave=False):
        rgb, ir, gt = rgb.to(device), ir.to(device), gt.to(device)
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss, parts = criterion(model(rgb, ir), gt)
        if train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        for k in COMPONENTS:
            totals[k] += parts[k]
    return {k: v / len(loader) for k, v in totals.items()}


def build_train_val(data_root: Path, object_names: list[str]):
    train_parts, val_parts = [], []
    for name in object_names:
        obj_root = data_root / name
        train_ds = RealDepthDataset(obj_root, augment=True)
        eval_ds = RealDepthDataset(obj_root, augment=False)
        train_idx, val_idx = sequential_split(len(train_ds), TRAIN_FRAC)
        train_parts.append(Subset(train_ds, train_idx))
        val_parts.append(Subset(eval_ds, val_idx))
        print(f"  {name}: train={len(train_idx)} val={len(val_idx)}")
    return ConcatDataset(train_parts), ConcatDataset(val_parts)


def main():
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/data/new_extracted_unzipped")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    test_objects, train_val_objects, groups = split_test_train_val(data_root, N_TEST_GROUPS)
    print(f"auto-discovered {sum(len(v) for v in groups.values())} recordings in {len(groups)} scene groups")
    print(f"  test group(s) (held out, all lighting variants): {test_objects}")
    print(f"  train/val recordings: {len(train_val_objects)}")

    print("\nbuilding train/val (pooled across recordings, sequential split per recording):")
    train_ds, val_ds = build_train_val(data_root, train_val_objects)
    print(f"total: train={len(train_ds)} val={len(val_ds)}")

    test_datasets = {name: RealDepthDataset(data_root / name, augment=False) for name in test_objects}
    for name, ds in test_datasets.items():
        print(f"held-out test recording '{name}': {len(ds)} frames (never trained on)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loaders = {name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for name, ds in test_datasets.items()}

    model = DepthCorrectionNet().to(device)
    ckpt = torch.load(PHASE1_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded {PHASE1_CKPT} (phase1 epoch {ckpt['epoch']}, val_total={ckpt['val_total']:.4f})")

    for p in model.rgb_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"frozen (rgb_encoder): {sum(p.numel() for p in model.rgb_encoder.parameters()):,} params | "
          f"trainable: {sum(p.numel() for p in trainable):,} params")

    criterion = DepthCorrectionLoss(**LOSS_WEIGHTS).to(device)
    optimizer = torch.optim.Adam(trainable, lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"\n--- Phase 2 fine-tuning: {EPOCHS} epochs --- weights: {LOSS_WEIGHTS}\n")

    history, best_val = [], float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_parts = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_parts = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()

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

    best_ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    print("\n--- final test evaluation on held-out test group (best.pt) ---")
    for name, loader in test_loaders.items():
        test_parts = run_epoch(model, loader, criterion, optimizer, device, train=False)
        print(f"  {name}: " + " ".join(f"{k}={test_parts[k]:.4f}" for k in COMPONENTS))

    print(f"\ncheckpoints saved to {CKPT_DIR.resolve()}")


if __name__ == "__main__":
    main()