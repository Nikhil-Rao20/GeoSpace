"""
config.py — All hyperparameters in one place.
Edit this file to change training settings.
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:

    # ── Paths ─────────────────────────────────────────────────────────────────
    base_dir: str = (
        "/mnt/e3dbc9b9-6856-470d-84b1-ff55921cd906"
        "/Datasets/IIT Tirupathi/preprocessed_dataset"
    )
    checkpoint_dir: str = "checkpoints"
    log_dir: str        = "logs_testing"
    cas_cache_file: str = "cas_weights_cache.json"   # cached inside base_dir

    # ── Data ──────────────────────────────────────────────────────────────────
    num_classes: int  = 7
    image_size: int   = 512
    num_workers: int  = 0
    pin_memory: bool  = True

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "nvidia/mit-b4"   # HuggingFace hub id

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs:    int   = 50
    batch_size:    int   = 8      # reduce to 8 if CUDA OOM
    learning_rate: float = 6e-5
    weight_decay:  float = 0.01
    grad_clip:     float = 1.0
    warmup_epochs: int   = 2

    # ── DRW (Deferred Re-Weighting) Schedule ──────────────────────────────────
    # Phase 1: epochs  0 … drw_start_epoch-1  → uniform weights, no boundary loss
    # Phase 2: epochs  drw_start_epoch … end  → class-balanced weights + boundary
    drw_start_epoch:      int = 25     # 50 % of num_epochs
    boundary_start_epoch: int = 25

    # ── Loss Combination Weights ───────────────────────────────────────────────
    lambda_focal_tversky: float = 0.5
    lambda_boundary:      float = 0.3
    lambda_topk_ce:       float = 0.2
    topk_ratio:           float = 0.10   # keep worst 10 % of pixels

    # ── Focal Tversky Params ──────────────────────────────────────────────────
    tversky_alpha: float = 0.3   # FP weight  (lower → penalise FP less)
    tversky_beta:  float = 0.7   # FN weight  (higher → penalise missed detections more)
    tversky_gamma: float = 0.75  # focal exponent (< 1 → focus on hard pixels)

    # ── Boundary Loss Params ──────────────────────────────────────────────────
    boundary_theta:  float = 5.0  # multiplier for boundary pixels
    boundary_kernel: int   = 5    # morphological kernel size (px)

    # ── Class Weights (Phase 2 of DRW) ────────────────────────────────────────
    # Derived from inverse pixel frequency; background always down-weighted.
    class_weights: List[float] = field(default_factory=lambda: [
        0.3,   # 0  Background     (~54 % of pixels — heavily suppressed)
        1.0,   # 1  Building       (~24 %)
        2.0,   # 2  Road           (~ 7 %)
        2.5,   # 3  Waterbody_poly (~ 6 %)
        10.0,  # 4  Waterbody_line (~ 0.02 % — very thin features)
        6.0,   # 5  Utility_poly   (~ 0.01 %)
        15.0,  # 6  Bridge         (~ 0.005 % — 17 patches total)
    ])

    # ── CAS (Class-Aware Sampling) ────────────────────────────────────────────
    cas_threshold:  float = 0.01
    cas_max_repeat: float = 15.0

    # ── ImageNet Normalisation ────────────────────────────────────────────────
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # ── Class Names (for logging & metrics) ───────────────────────────────────
    class_names: List[str] = field(default_factory=lambda: [
        "Background",
        "Building",
        "Road",
        "Waterbody_poly",
        "Waterbody_line",
        "Utility_poly",
        "Bridge",
    ])

    # ── cRT (Classifier Re-Training) ─────────────────────────────────────────
    crt_epochs: int   = 15
    crt_lr:     float = 6e-6   # 10× smaller than main LR
