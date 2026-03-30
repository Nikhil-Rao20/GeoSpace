"""
train.py — Main training + cRT script for SVAMITVA Stage-1 segmentation.

Usage:
    # Full run: main training → cRT
    python src/train.py

    # Main training only (skip cRT):
    python src/train.py --no-crt

    # cRT only (provide existing checkpoint):
    python src/train.py --crt-only --main-ckpt checkpoints_eswar/segformer_b4_main_best.pth
"""

import os
import sys
import json
import time
import logging
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR
)
from tqdm import tqdm

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config  import Config
from dataset import SVAMITVADataset, build_cas_sampler, get_train_transforms, get_val_transforms
from losses  import CombinedLoss
from model   import SVAMITVASegFormer
from metrics import SegmentationMetrics


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


# ── One Training Epoch ────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch: int,
    logger,
) -> dict:
    model.train()
    criterion.set_epoch(epoch)

    total    = 0.0
    ft_sum   = 0.0
    bd_sum   = 0.0
    topk_sum = 0.0
    n        = 0

    bar = tqdm(loader, desc=f"E{epoch+1:03d}[Train]", leave=False, ncols=100)
    for imgs, labels in bar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            logits    = model(imgs)
            loss_dict = criterion(logits, labels)
            loss      = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total    += loss.item()
        ft_sum   += loss_dict["focal_tversky"].item()
        bd_sum   += loss_dict["boundary"].item()
        topk_sum += loss_dict["topk_ce"].item()
        n        += 1

        bar.set_postfix(
            loss=f"{loss.item():.4f}",
            ph=loss_dict["phase"],
        )

    return {
        "total":         total    / n,
        "focal_tversky": ft_sum   / n,
        "boundary":      bd_sum   / n,
        "topk_ce":       topk_sum / n,
    }


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    metrics: SegmentationMetrics,
) -> dict:
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n          = 0

    for imgs, labels in tqdm(loader, desc="[Val]", leave=False, ncols=100):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda"):
            logits    = model(imgs)
            loss_dict = criterion(logits, labels)

        metrics.update(logits, labels)
        total_loss += loss_dict["total"].item()
        n          += 1

    val_m          = metrics.compute()
    val_m["loss"]  = total_loss / n
    return val_m


# ── Main Training ─────────────────────────────────────────────────────────────

def run_training(cfg: Config) -> str:
    """
    Trains Stage-1 SegFormer-B4 model.
    Returns path to the best checkpoint.
    """
    logger = setup_logger(cfg.log_dir, "main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"AMP: enabled")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_file = os.path.join(cfg.base_dir, "train.txt")
    val_file   = os.path.join(cfg.base_dir, "val.txt")

    train_ds = SVAMITVADataset(
        cfg.base_dir, train_file, transform=get_train_transforms(cfg.mean, cfg.std)
    )
    val_ds = SVAMITVADataset(
        cfg.base_dir, val_file, transform=get_val_transforms(cfg.mean, cfg.std)
    )
    logger.info(f"Train: {len(train_ds):,} patches | Val: {len(val_ds):,} patches")

    # ── CAS Sampler ───────────────────────────────────────────────────────
    cache_path  = os.path.join(cfg.base_dir, cfg.cas_cache_file)
    cas_sampler = build_cas_sampler(
        base_dir      = cfg.base_dir,
        train_patches = train_ds.patches,
        num_classes   = cfg.num_classes,
        threshold     = cfg.cas_threshold,
        max_repeat    = cfg.cas_max_repeat,
        cache_path    = cache_path,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.batch_size,
        sampler     = cas_sampler,
        num_workers = cfg.num_workers,
        pin_memory  = cfg.pin_memory,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = cfg.pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = SVAMITVASegFormer(
        num_classes = cfg.num_classes,
        pretrained  = cfg.model_name,
    ).to(device)
    model.print_param_summary()

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = CombinedLoss(
        num_classes          = cfg.num_classes,
        class_weights        = cfg.class_weights,
        lambda_ft            = cfg.lambda_focal_tversky,
        lambda_bd            = cfg.lambda_boundary,
        lambda_topk          = cfg.lambda_topk_ce,
        topk_ratio           = cfg.topk_ratio,
        tversky_alpha        = cfg.tversky_alpha,
        tversky_beta         = cfg.tversky_beta,
        tversky_gamma        = cfg.tversky_gamma,
        boundary_theta       = cfg.boundary_theta,
        boundary_kernel      = cfg.boundary_kernel,
        drw_start_epoch      = cfg.drw_start_epoch,
        boundary_start_epoch = cfg.boundary_start_epoch,
    ).to(device)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
    )

    warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max   = cfg.num_epochs - cfg.warmup_epochs,
        eta_min = 1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [cfg.warmup_epochs],
    )

    scaler  = GradScaler()
    metrics = SegmentationMetrics(cfg.num_classes, cfg.class_names)

    best_miou    = 0.0
    best_ckpt    = os.path.join(cfg.checkpoint_dir, "segformer_b4_main_best.pth")
    last_ckpt    = os.path.join(cfg.checkpoint_dir, "segformer_b4_main_last.pth")
    history_path = os.path.join(cfg.log_dir, "main_history.json")
    history      = []

    logger.info("=" * 70)
    logger.info("STARTING MAIN TRAINING")
    logger.info(f"  Epochs       : {cfg.num_epochs}")
    logger.info(f"  Batch size   : {cfg.batch_size}")
    logger.info(f"  LR           : {cfg.learning_rate}")
    logger.info(f"  DRW switch   : epoch {cfg.drw_start_epoch}")
    logger.info(f"  Boundary     : enabled from epoch {cfg.boundary_start_epoch}")
    logger.info("=" * 70)

    for epoch in range(cfg.num_epochs):
        t0 = time.time()

        # Announce DRW phase change
        if epoch == cfg.drw_start_epoch:
            logger.info("=" * 70)
            logger.info(">>> DRW PHASE 2: class-balanced weights + boundary loss active")
            logger.info("=" * 70)

        # ── Train ─────────────────────────────────────────────────────────
        train_s = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, logger
        )

        # ── Validate ──────────────────────────────────────────────────────
        val_s = validate(model, val_loader, criterion, device, metrics)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        logger.info(
            f"E{epoch+1:03d}/{cfg.num_epochs} | "
            f"lr={lr_now:.2e} | "
            f"train={train_s['total']:.4f} "
            f"(ft={train_s['focal_tversky']:.3f} "
            f"bd={train_s['boundary']:.3f} "
            f"tk={train_s['topk_ce']:.3f}) | "
            f"val_loss={val_s['loss']:.4f} | "
            f"mIoU_fg={val_s['mean_iou_fg']:.4f} | "
            f"{elapsed:.0f}s"
        )

        # Per-class breakdown every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info("Per-class metrics:\n" + metrics.format_summary())

        # ── Save best ─────────────────────────────────────────────────────
        if val_s["mean_iou_fg"] > best_miou:
            best_miou = val_s["mean_iou_fg"]
            save_checkpoint(
                {
                    "epoch":          epoch,
                    "model_state":    model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_miou":      best_miou,
                    "val_metrics":    val_s,
                    "config":         cfg.__dict__,
                    "stage":          "main",
                },
                best_ckpt,
            )
            logger.info(f"  ✓ Best mIoU(fg): {best_miou:.4f}  → {best_ckpt}")

        # ── Save last ─────────────────────────────────────────────────────
        save_checkpoint(
            {
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_miou":      best_miou,
                "config":         cfg.__dict__,
                "stage":          "main",
            },
            last_ckpt,
        )

        # ── History ───────────────────────────────────────────────────────
        history.append({"epoch": epoch + 1, "train": train_s, "val": val_s})
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    logger.info(f"Training done. Best mIoU(fg): {best_miou:.4f}  →  {best_ckpt}")
    return best_ckpt


# ── cRT ───────────────────────────────────────────────────────────────────────

def run_crt(cfg: Config, main_ckpt_path: str) -> str:
    """
    Classifier Re-Training (cRT):
        1. Load best main checkpoint
        2. Freeze MiT-B4 encoder
        3. Re-train decode head for cfg.crt_epochs with class-balanced loss
        4. Save separately as segformer_b4_crt_best.pth

    Returns path to best cRT checkpoint.

    Reference (concept from two-stage decoupled training):
        Kang et al., "Decoupling Representation and Classifier for
        Long-Tailed Recognition", ICLR 2020.
    """
    logger = setup_logger(cfg.log_dir, "crt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load ──────────────────────────────────────────────────────────────
    logger.info(f"[cRT] Loading checkpoint: {main_ckpt_path}")
    model = SVAMITVASegFormer(
        num_classes = cfg.num_classes,
        pretrained  = cfg.model_name,
    )
    ckpt = torch.load(main_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    logger.info(f"[cRT] Resumed from epoch {ckpt['epoch']+1}, best mIoU: {ckpt['best_miou']:.4f}")

    # ── Freeze encoder ────────────────────────────────────────────────────
    model.freeze_encoder()

    # ── Data ──────────────────────────────────────────────────────────────
    train_file = os.path.join(cfg.base_dir, "train.txt")
    val_file   = os.path.join(cfg.base_dir, "val.txt")

    train_ds = SVAMITVADataset(
        cfg.base_dir, train_file, transform=get_train_transforms(cfg.mean, cfg.std)
    )
    val_ds = SVAMITVADataset(
        cfg.base_dir, val_file, transform=get_val_transforms(cfg.mean, cfg.std)
    )

    # Re-use cached CAS weights
    cache_path  = os.path.join(cfg.base_dir, cfg.cas_cache_file)
    cas_sampler = build_cas_sampler(
        base_dir      = cfg.base_dir,
        train_patches = train_ds.patches,
        num_classes   = cfg.num_classes,
        threshold     = cfg.cas_threshold,
        max_repeat    = cfg.cas_max_repeat,
        cache_path    = cache_path,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.batch_size,
        sampler     = cas_sampler,
        num_workers = cfg.num_workers,
        pin_memory  = cfg.pin_memory,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = cfg.pin_memory,
    )

    # ── Loss — Phase 2 from epoch 0 ───────────────────────────────────────
    criterion = CombinedLoss(
        num_classes          = cfg.num_classes,
        class_weights        = cfg.class_weights,
        lambda_ft            = cfg.lambda_focal_tversky,
        lambda_bd            = cfg.lambda_boundary,
        lambda_topk          = cfg.lambda_topk_ce,
        topk_ratio           = cfg.topk_ratio,
        tversky_alpha        = cfg.tversky_alpha,
        tversky_beta         = cfg.tversky_beta,
        tversky_gamma        = cfg.tversky_gamma,
        boundary_theta       = cfg.boundary_theta,
        boundary_kernel      = cfg.boundary_kernel,
        drw_start_epoch      = 0,   # class-balanced from the start
        boundary_start_epoch = 0,
    ).to(device)

    # ── Optimizer — only decode head, 10× smaller LR ─────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg.crt_lr,
        weight_decay = cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.crt_epochs, eta_min=1e-8)
    scaler    = GradScaler()
    metrics   = SegmentationMetrics(cfg.num_classes, cfg.class_names)

    best_miou    = 0.0
    best_ckpt    = os.path.join(cfg.checkpoint_dir, "segformer_b4_crt_best.pth")
    history_path = os.path.join(cfg.log_dir, "crt_history.json")
    history      = []

    logger.info("=" * 70)
    logger.info("STARTING cRT (Classifier Re-Training)")
    logger.info(f"  cRT epochs: {cfg.crt_epochs}  |  LR: {cfg.crt_lr}")
    logger.info("=" * 70)

    for epoch in range(cfg.crt_epochs):
        t0 = time.time()

        train_s = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, logger
        )
        val_s   = validate(model, val_loader, criterion, device, metrics)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"[cRT] E{epoch+1:02d}/{cfg.crt_epochs} | "
            f"train={train_s['total']:.4f} | "
            f"val_loss={val_s['loss']:.4f} | "
            f"mIoU_fg={val_s['mean_iou_fg']:.4f} | "
            f"{elapsed:.0f}s"
        )

        if (epoch + 1) % 5 == 0:
            logger.info("[cRT] Per-class metrics:\n" + metrics.format_summary())

        if val_s["mean_iou_fg"] > best_miou:
            best_miou = val_s["mean_iou_fg"]
            save_checkpoint(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "best_miou":   best_miou,
                    "val_metrics": val_s,
                    "config":      cfg.__dict__,
                    "stage":       "crt",
                },
                best_ckpt,
            )
            logger.info(f"  ✓ [cRT] Best mIoU(fg): {best_miou:.4f}  → {best_ckpt}")

        history.append({"epoch": epoch + 1, "train": train_s, "val": val_s})
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    logger.info(f"[cRT] Done. Best mIoU(fg): {best_miou:.4f}  →  {best_ckpt}")
    return best_ckpt


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SVAMITVA Stage-1 Training")
    p.add_argument(
        "--no-crt", action="store_true",
        help="Skip cRT after main training",
    )
    p.add_argument(
        "--crt-only", action="store_true",
        help="Run cRT only (requires --main-ckpt)",
    )
    p.add_argument(
        "--main-ckpt", type=str, default=None,
        help="Path to existing main checkpoint (for --crt-only)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = Config()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    if args.crt_only:
        assert args.main_ckpt and os.path.exists(args.main_ckpt), \
            f"--main-ckpt not found: {args.main_ckpt}"
        run_crt(cfg, args.main_ckpt)

    else:
        best_main = run_training(cfg)

        if not args.no_crt:
            run_crt(cfg, best_main)
