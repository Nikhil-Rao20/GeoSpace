"""
losses.py — Loss functions for SVAMITVA segmentation.

Components:
    FocalTverskyLoss  — primary region loss, penalises FN heavily
    BoundaryLoss      — focuses on boundary pixels (pure PyTorch, no scipy)
    TopKCELoss        — hard-pixel mining via top-k cross entropy
    CombinedLoss      — ties all three together with DRW scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ── Focal Tversky Loss ────────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """
    Multi-class Focal Tversky Loss.

    Tversky index generalises Dice:
        TI_c = TP / (TP + alpha*FP + beta*FN)

    Setting alpha < 0.5 and beta > 0.5 penalises false negatives more than
    false positives — essential for small/rare foreground classes.

    The focal exponent gamma < 1 shifts focus to hard, misclassified pixels.

    Reference:
        Abraham & Khan, "A Novel Focal Tversky loss function with improved
        Attention U-Net for lesion segmentation", ISBI 2019.
    """

    def __init__(
        self,
        alpha:         float            = 0.3,
        beta:          float            = 0.7,
        gamma:         float            = 0.75,
        num_classes:   int              = 7,
        class_weights: Optional[List[float]] = None,
        smooth:        float            = 1e-6,
    ):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            # Normalise so weights sum to num_classes (preserves scale)
            w = w / w.sum() * num_classes
        else:
            w = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("class_weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred  : (B, C, H, W)  raw logits
        target: (B, H, W)     integer class indices in [0, C-1]
        """
        B, C, H, W = pred.shape

        pred_soft  = F.softmax(pred, dim=1)                          # (B, C, H, W)
        target_1h  = (
            F.one_hot(target, C).permute(0, 3, 1, 2).float()
        )                                                             # (B, C, H, W)

        # Sum over batch + spatial → per-class scalars
        tp = (pred_soft *       target_1h ).sum(dim=(0, 2, 3))       # (C,)
        fp = (pred_soft * (1 - target_1h)).sum(dim=(0, 2, 3))        # (C,)
        fn = ((1 - pred_soft) * target_1h).sum(dim=(0, 2, 3))        # (C,)

        tversky      = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )                                                             # (C,)
        focal_tv     = (1 - tversky) ** self.gamma                   # (C,)
        weighted     = focal_tv * self.class_weights                 # (C,)

        return weighted.mean()


# ── Boundary Loss ─────────────────────────────────────────────────────────────

class BoundaryLoss(nn.Module):
    """
    Boundary-aware cross-entropy loss.

    Boundary pixels are detected via a morphological gradient:
        boundary = dilation(mask) - erosion(mask) > threshold

    These pixels receive weight `theta` in the per-pixel CE map;
    interior and background pixels receive weight 1.

    This is implemented with PyTorch's max_pool2d as a dilation proxy
    and 1 - max_pool2d(1-mask) as erosion — no scipy required.

    Reference (adapted for multi-class):
        Kervadec et al., "Boundary loss for highly unbalanced segmentation",
        MIDL 2019.
    """

    def __init__(
        self,
        num_classes:   int              = 7,
        theta:         float            = 5.0,
        kernel_size:   int              = 5,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.theta       = theta
        self.kernel_size = kernel_size

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
        else:
            w = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("class_weights", w)

    def _boundary_mask(self, target: torch.Tensor) -> torch.Tensor:
        """
        target : (B, H, W) long
        returns: (B, H, W) float — 1.0 at boundary pixels, 0.0 elsewhere
        """
        k   = self.kernel_size
        pad = k // 2
        B, H, W = target.shape
        boundary = torch.zeros(B, H, W, device=target.device, dtype=torch.float32)

        # One-hot over foreground classes only
        t1h = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        for c in range(1, self.num_classes):            # skip background (c=0)
            ch       = t1h[:, c : c + 1, :, :]         # (B, 1, H, W)
            dilated  = F.max_pool2d(ch,       k, stride=1, padding=pad)
            eroded   = 1.0 - F.max_pool2d(1.0 - ch, k, stride=1, padding=pad)
            bd_c     = ((dilated - eroded) > 0.5).squeeze(1).float()   # (B, H, W)
            boundary = torch.clamp(boundary + bd_c, max=1.0)

        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred  : (B, C, H, W) raw logits
        target: (B, H, W)    integer class indices
        """
        boundary   = self._boundary_mask(target)              # (B, H, W)
        weight_map = 1.0 + (self.theta - 1.0) * boundary     # (B, H, W)

        ce = F.cross_entropy(
            pred, target,
            weight=self.class_weights,
            reduction="none",
        )                                                      # (B, H, W)

        return (ce * weight_map).mean()


# ── TopK CE Loss ──────────────────────────────────────────────────────────────

class TopKCELoss(nn.Module):
    """
    Cross-entropy computed only on the hardest k% of pixels per batch.

    Automatically performs implicit hard-pixel mining without needing
    explicit sample selection. Complements Tversky (which focuses on
    class-level imbalance) by focusing on pixel-level difficulty.

    k = 0.10 → keep the worst 10 % of pixels by CE value.

    Reference:
        Wu et al., "Wider or Deeper: Revisiting the ResNet Model for
        Visual Recognition", PR 2019.
    """

    def __init__(
        self,
        k:             float            = 0.10,
        class_weights: Optional[List[float]] = None,
        num_classes:   int              = 7,
    ):
        super().__init__()
        self.k = k

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
        else:
            w = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("class_weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred  : (B, C, H, W) raw logits
        target: (B, H, W)    integer class indices
        """
        ce = F.cross_entropy(
            pred, target,
            weight=self.class_weights,
            reduction="none",
        ).reshape(-1)                                  # (B*H*W,)

        k_px      = max(1, int(self.k * ce.numel()))
        topk, _   = torch.topk(ce, k_px)
        return topk.mean()


# ── Combined Loss with DRW ────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Main loss function combining FocalTversky + BoundaryLoss + TopKCE.

    DRW Schedule
    ────────────
    Phase 1  (epoch < drw_start_epoch):
        • Uniform class weights (background down-weighted at 0.3)
        • Boundary loss DISABLED (too noisy before representations stabilise)
        • L = λ_ft × FocalTversky + λ_tk × TopKCE

    Phase 2  (epoch ≥ drw_start_epoch):
        • Class-balanced weights (from config — rare classes heavily weighted)
        • Boundary loss ENABLED
        • L = λ_ft × FocalTversky + λ_bd × BoundaryLoss + λ_tk × TopKCE

    Usage:
        criterion = CombinedLoss(...)
        criterion.set_epoch(epoch)          # call at start of each epoch
        loss_dict = criterion(logits, labels)
        loss_dict["total"].backward()
    """

    def __init__(
        self,
        num_classes:          int              = 7,
        class_weights:        Optional[List[float]] = None,
        lambda_ft:            float            = 0.5,
        lambda_bd:            float            = 0.3,
        lambda_topk:          float            = 0.2,
        topk_ratio:           float            = 0.10,
        tversky_alpha:        float            = 0.3,
        tversky_beta:         float            = 0.7,
        tversky_gamma:        float            = 0.75,
        boundary_theta:       float            = 5.0,
        boundary_kernel:      int              = 5,
        drw_start_epoch:      int              = 25,
        boundary_start_epoch: int              = 25,
    ):
        super().__init__()
        self.lambda_ft            = lambda_ft
        self.lambda_bd            = lambda_bd
        self.lambda_topk          = lambda_topk
        self.drw_start_epoch      = drw_start_epoch
        self.boundary_start_epoch = boundary_start_epoch
        self.current_epoch        = 0

        # Phase 1 weights: background at 0.3, everything else at 1.0
        uniform_w        = [1.0] * num_classes
        uniform_w[0]     = 0.3

        # ── Phase 1 losses (uniform weights) ─────────────────────────────
        self.ft_p1   = FocalTverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma,
            num_classes=num_classes, class_weights=uniform_w,
        )
        self.topk_p1 = TopKCELoss(
            k=topk_ratio, class_weights=uniform_w, num_classes=num_classes,
        )

        # ── Phase 2 losses (class-balanced weights) ───────────────────────
        self.ft_p2       = FocalTverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma,
            num_classes=num_classes, class_weights=class_weights,
        )
        self.boundary    = BoundaryLoss(
            num_classes=num_classes,
            theta=boundary_theta,
            kernel_size=boundary_kernel,
            class_weights=class_weights,
        )
        self.topk_p2     = TopKCELoss(
            k=topk_ratio, class_weights=class_weights, num_classes=num_classes,
        )

    def set_epoch(self, epoch: int):
        """Must be called at the start of each epoch."""
        self.current_epoch = epoch

    @property
    def phase(self) -> int:
        return 2 if self.current_epoch >= self.drw_start_epoch else 1

    @property
    def boundary_active(self) -> bool:
        return self.current_epoch >= self.boundary_start_epoch

    def forward(
        self,
        pred:   torch.Tensor,   # (B, C, H, W)
        target: torch.Tensor,   # (B, H, W) long
    ) -> dict:
        """
        Returns a dict with keys:
            total           — scalar loss to call .backward() on
            focal_tversky   — detached component value (for logging)
            boundary        — detached component value
            topk_ce         — detached component value
            phase           — int 1 or 2
        """
        if self.phase == 2:
            ft_loss   = self.ft_p2(pred, target)
            topk_loss = self.topk_p2(pred, target)
        else:
            ft_loss   = self.ft_p1(pred, target)
            topk_loss = self.topk_p1(pred, target)

        if self.boundary_active:
            bd_loss = self.boundary(pred, target)
            total   = (
                self.lambda_ft   * ft_loss
                + self.lambda_bd   * bd_loss
                + self.lambda_topk * topk_loss
            )
        else:
            bd_loss = torch.zeros(1, device=pred.device)
            total   = self.lambda_ft * ft_loss + self.lambda_topk * topk_loss

        return {
            "total":         total,
            "focal_tversky": ft_loss.detach(),
            "boundary":      bd_loss.detach(),
            "topk_ce":       topk_loss.detach(),
            "phase":         self.phase,
        }
