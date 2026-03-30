"""
stage2_train.py — Stage-2 subtype classifier training.

Usage:
    # Extract ROIs (one-time, ~10 min) then train all 3 classifiers:
    python src/stage2_train.py

    # Extract only:
    python src/stage2_train.py --extract-only

    # Train only (ROIs already extracted):
    python src/stage2_train.py --train-only

    # Train a single classifier:
    python src/stage2_train.py --train-only --cls building
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))

from config          import Config
from stage2_dataset  import (
    ROIExtractor, ClassifierDataset,
    CLASSIFIER_CONFIG,
    get_cls_train_transforms, get_cls_val_transforms,
    build_classifier_sampler, split_records_by_village,
)
from stage2_losses   import LDAMDRWLoss
from stage2_model    import SubtypeClassifier


# ── Per-classifier hyperparameters ───────────────────────────────────────────

# Updated after full-dataset scan (Stage-1 results confirmed these counts)
CLS_COUNTS = {
    "building":  [8611, 797, 6281, 1702],   # RCC, Tiled, Tin, Others
    "road":      [694, 964, 5178, 1518],    # code 2,3,4,5
    "waterbody": [1347, 194, 228, 288],     # code4, code1, code2, other(3+5+6)
}

# Note: these are patch-level counts; actual crop counts will be higher
# (multiple components per patch). LDAM-DRW uses them for margin computation
# and will be recalculated from actual crop counts after extraction.

CLS_EPOCHS    = {"building": 40, "road": 40, "waterbody": 40}
CLS_LR        = {"building": 1e-4, "road": 1e-4, "waterbody": 1e-4}
CLS_BATCH     = {"building": 64, "road": 64, "waterbody": 64}
CLS_BACKBONE  = {"building": "EfficientNet-B2", "road": "EfficientNet-B0",
                 "waterbody": "EfficientNet-B0"}


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh  = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    ch  = logging.StreamHandler()
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ── Train one classifier ──────────────────────────────────────────────────────

def train_classifier(cls_name: str, cfg: Config):
    logger = setup_logger(cfg.log_dir, f"stage2_{cls_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crops_dir  = os.path.join(cfg.base_dir, "stage2_crops")
    rec_path   = os.path.join(crops_dir, f"{cls_name}_records.json")
    cls_cfg    = CLASSIFIER_CONFIG[cls_name]
    num_classes = cls_cfg["num_classes"]
    class_names = cls_cfg["class_names"]

    # ── Load records ──────────────────────────────────────────────────────
    with open(rec_path) as f:
        all_records = json.load(f)
    logger.info(f"[{cls_name}] Total crops: {len(all_records):,}")

    # Class counts from actual extracted crops
    actual_counts = [0] * num_classes
    for r in all_records:
        actual_counts[r["label"]] += 1
    logger.info(f"[{cls_name}] Crop class counts: {actual_counts}")
    logger.info(f"[{cls_name}] Classes: {class_names}")

    # ── Train / Val split ─────────────────────────────────────────────────
    train_recs, val_recs = split_records_by_village(all_records, val_frac=0.20)
    logger.info(f"[{cls_name}] Train: {len(train_recs):,}  Val: {len(val_recs):,}")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = ClassifierDataset(
        crops_dir, cls_name, records=train_recs,
        transform=get_cls_train_transforms(
            size=cls_cfg["crop_size"], mean=cfg.mean, std=cfg.std
        ),
    )
    val_ds = ClassifierDataset(
        crops_dir, cls_name, records=val_recs,
        transform=get_cls_val_transforms(mean=cfg.mean, std=cfg.std),
    )

    # ── Sampler ───────────────────────────────────────────────────────────
    sampler = build_classifier_sampler(train_recs, num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=CLS_BATCH[cls_name],
        sampler=sampler, num_workers=4,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CLS_BATCH[cls_name],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = SubtypeClassifier(
        cls_name=cls_name, num_classes=num_classes
    ).to(device)
    logger.info(
        f"[{cls_name}] {CLS_BACKBONE[cls_name]}  "
        f"| params: {model.get_trainable_params()/1e6:.1f}M"
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    # Recompute train-split counts for accurate LDAM margins
    train_counts = [0] * num_classes
    for r in train_recs:
        train_counts[r["label"]] += 1
    # Floor at 1 to avoid division by zero for unseen classes
    train_counts = [max(1, c) for c in train_counts]

    total_eps = CLS_EPOCHS[cls_name]
    criterion = LDAMDRWLoss(
        cls_counts   = train_counts,
        max_margin   = 0.5,
        drw_start    = 0.60,
        total_epochs = total_eps,
    ).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=CLS_LR[cls_name], weight_decay=1e-4)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
    cosine = CosineAnnealingLR(optimizer, T_max=total_eps - 3, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])
    scaler    = GradScaler()

    best_acc   = 0.0
    best_ckpt  = os.path.join(cfg.checkpoint_dir, f"classifier_{cls_name}.pth")
    hist_path  = os.path.join(cfg.log_dir, f"stage2_{cls_name}_history.json")
    history    = []

    logger.info("=" * 65)
    logger.info(f"TRAINING STAGE-2 CLASSIFIER: {cls_name.upper()}")
    logger.info(f"  Epochs: {total_eps}  |  LR: {CLS_LR[cls_name]}  |  "
                f"Batch: {CLS_BATCH[cls_name]}  |  DRW at: "
                f"{criterion.drw_epoch}")
    logger.info("=" * 65)

    for epoch in range(total_eps):
        t0 = time.time()
        criterion.set_epoch(epoch)

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0; n = 0
        for imgs, labels in tqdm(train_loader,
                                  desc=f"[{cls_name}] E{epoch+1:03d}",
                                  leave=False, ncols=90):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item(); n += 1
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        all_preds  = []; all_labels = []; val_loss = 0.0; nv = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type="cuda"):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_loss += loss.item(); nv += 1

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc        = (all_preds == all_labels).mean()
        elapsed    = time.time() - t0

        logger.info(
            f"[{cls_name}] E{epoch+1:03d}/{total_eps} | "
            f"ph={criterion.phase} | "
            f"train={train_loss/n:.4f} | "
            f"val_loss={val_loss/nv:.4f} | "
            f"acc={acc:.4f} | "
            f"{elapsed:.0f}s"
        )

        # Detailed report every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == total_eps - 1:
            report = classification_report(
                all_labels, all_preds,
                target_names=class_names,
                digits=4, zero_division=0,
            )
            logger.info(f"\n{report}")
            cm = confusion_matrix(all_labels, all_preds)
            logger.info(f"Confusion matrix:\n{cm}")

        # ── Save best ─────────────────────────────────────────────────────
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "best_acc":    best_acc,
                "cls_name":    cls_name,
                "class_names": class_names,
                "num_classes": num_classes,
            }, best_ckpt)
            logger.info(f"  ✓ Best acc: {best_acc:.4f}  → {best_ckpt}")

        history.append({"epoch": epoch+1, "train_loss": train_loss/n,
                         "val_loss": val_loss/nv, "val_acc": acc})
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    logger.info(f"[{cls_name}] Done. Best acc: {best_acc:.4f}  →  {best_ckpt}")
    return best_ckpt


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 Subtype Classifier Training")
    p.add_argument("--extract-only", action="store_true",
                   help="Only run ROI extraction, do not train")
    p.add_argument("--train-only", action="store_true",
                   help="Skip extraction, train from existing crops")
    p.add_argument("--cls", type=str, default=None,
                   choices=["building", "road", "waterbody"],
                   help="Train a single classifier only")
    p.add_argument("--force-extract", action="store_true",
                   help="Force re-extraction even if cache exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = Config()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    crops_dir = os.path.join(cfg.base_dir, "stage2_crops")

    # ── Step 1: ROI Extraction ────────────────────────────────────────────
    if not args.train_only:
        print("\n" + "="*60)
        print("STAGE-2  STEP 1: ROI EXTRACTION")
        print("="*60)
        extractor = ROIExtractor(
            base_dir   = cfg.base_dir,
            crops_dir  = crops_dir,
            split_file = os.path.join(cfg.base_dir, "train.txt"),
        )
        extractor.extract_all(force=args.force_extract)

    if args.extract_only:
        print("\nExtraction complete. Run with --train-only to start training.")
        exit(0)

    # ── Step 2: Train classifiers ─────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE-2  STEP 2: CLASSIFIER TRAINING")
    print("="*60)

    cls_to_train = [args.cls] if args.cls else ["building", "road", "waterbody"]

    for cls_name in cls_to_train:
        print(f"\n{'─'*60}")
        print(f"  Training: {cls_name.upper()}")
        print(f"{'─'*60}")
        train_classifier(cls_name, cfg)

    print("\n" + "="*60)
    print("STAGE-2 COMPLETE")
    print(f"  Checkpoints saved in: {cfg.checkpoint_dir}/")
    print("="*60)