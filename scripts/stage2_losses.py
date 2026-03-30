"""
stage2_losses.py — LDAM-DRW loss for Stage-2 imbalanced classification.

LDAM (Label-Distribution-Aware Margin) enforces larger decision margins
for minority classes. DRW (Deferred Re-Weighting) applies class-balanced
weights only after the model has stabilised (60% of training epochs).

Reference:
    Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware
    Margin Loss", NeurIPS 2019.

    Sulake (RGUKT Nuzvid), "Loss Design and Architecture Selection for
    Long-Tailed Multi-Label Chest X-Ray Classification", arXiv:2603.02294.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LDAMLoss(nn.Module):
    """
    LDAM loss with optional class-balanced re-weighting (DRW).

    Each class c has a margin:
        Δ_c = C / n_c^{1/4}
    where n_c is the training sample count and C is a scale constant.

    The per-sample logit for its true class is shifted down by Δ_c before
    computing cross-entropy, forcing the model to learn a larger margin for
    minority classes.

    Args:
        cls_counts:   list of training sample counts per class  [n_0, n_1, ...]
        max_margin:   margin scale constant C (default 0.5)
        cls_weights:  per-class CE weights for DRW phase (None = uniform)
    """

    def __init__(
        self,
        cls_counts:  List[int],
        max_margin:  float           = 0.5,
        cls_weights: List[float]     = None,
    ):
        super().__init__()
        self.max_margin = max_margin
        counts = np.array(cls_counts, dtype=np.float32)

        # Margin per class
        margins = max_margin / (counts ** 0.25)
        margins = margins / margins.max() * max_margin   # normalise
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))

        if cls_weights is not None:
            w = torch.tensor(cls_weights, dtype=torch.float32)
        else:
            w = torch.ones(len(cls_counts), dtype=torch.float32)
        self.register_buffer("cls_weights", w)

    def forward(
        self,
        logits:  torch.Tensor,   # (B, C)  — may be float16 under AMP autocast
        targets: torch.Tensor,   # (B,) long
        use_weights: bool = True,
    ) -> torch.Tensor:
        # Always compute loss in float32 — standard AMP pattern for loss fns.
        # GradScaler handles the float16 ↔ float32 conversion transparently.
        logits_f32 = logits.float()                          # (B, C) float32

        # Shift logit of true class down by its class-margin
        batch_margins = self.margins[targets]                # (B,) float32
        logits_m      = logits_f32.clone()
        logits_m.scatter_(
            1,
            targets.unsqueeze(1),
            logits_f32.gather(1, targets.unsqueeze(1)) - batch_margins.unsqueeze(1),
        )

        # cls_weights buffer is float32 — matches logits_m, no cast needed
        w = self.cls_weights if use_weights else None
        return F.cross_entropy(logits_m, targets, weight=w)


class LDAMDRWLoss(nn.Module):
    """
    LDAM with Deferred Re-Weighting (DRW).

    Phase 1 (epoch < drw_start): LDAM margin, uniform weights.
    Phase 2 (epoch >= drw_start): LDAM margin, class-balanced weights.

    Usage:
        criterion = LDAMDRWLoss(cls_counts=[8611, 797, 6281, 1702], ...)
        criterion.set_epoch(epoch)
        loss = criterion(logits, targets)
    """

    def __init__(
        self,
        cls_counts:   List[int],
        max_margin:   float = 0.5,
        drw_start:    float = 0.60,   # fraction of total epochs to start DRW
        total_epochs: int   = 40,
    ):
        super().__init__()
        self.drw_epoch  = int(drw_start * total_epochs)
        self.epoch      = 0

        counts = np.array(cls_counts, dtype=np.float32)
        # Class-balanced weights: effective number of samples
        beta     = 0.9999
        eff_num  = (1.0 - beta ** counts) / (1.0 - beta)
        cb_w     = (1.0 / eff_num)
        cb_w     = cb_w / cb_w.sum() * len(cls_counts)   # normalise

        self.loss_p1 = LDAMLoss(
            cls_counts  = cls_counts,
            max_margin  = max_margin,
            cls_weights = None,                                    # uniform
        )
        self.loss_p2 = LDAMLoss(
            cls_counts  = cls_counts,
            max_margin  = max_margin,
            cls_weights = cb_w.tolist(),                           # balanced
        )

        print(f"[LDAM-DRW] DRW activates at epoch {self.drw_epoch}")
        print(f"[LDAM-DRW] Class-balanced weights: "
              f"{np.round(cb_w, 3).tolist()}")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @property
    def phase(self) -> int:
        return 2 if self.epoch >= self.drw_epoch else 1

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.phase == 2:
            return self.loss_p2(logits, targets, use_weights=True)
        return self.loss_p1(logits, targets, use_weights=False)
