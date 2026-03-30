"""
metrics.py — Segmentation metrics computed from a confusion matrix.

SegmentationMetrics accumulates predictions across batches (streaming),
then computes per-class IoU, precision, recall, F1, and mean IoU.
"""

import numpy as np
import torch
from typing import List


class SegmentationMetrics:
    """
    Accumulates a confusion matrix across batches and computes:

        • Per-class IoU  (Intersection over Union)
        • Per-class Precision, Recall, F1
        • Mean IoU — all classes  (including background)
        • Mean IoU — foreground only  (classes 1 … C-1)
        • Overall pixel accuracy

    Usage:
        metrics = SegmentationMetrics(num_classes=7, class_names=[...])
        for batch in val_loader:
            logits, labels = model(imgs), labels
            metrics.update(logits, labels)
        results = metrics.compute()
        metrics.reset()        # before next epoch
    """

    def __init__(
        self,
        num_classes:  int,
        class_names:  List[str],
        ignore_index: int = 255,
    ):
        self.num_classes  = num_classes
        self.class_names  = class_names
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Zero the confusion matrix."""
        self.confusion = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred  : (B, C, H, W) logits  OR  (B, H, W) class indices (long)
            target: (B, H, W)  ground-truth class indices (long)
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)        # (B, H, W)

        pred_np   = pred.cpu().numpy().astype(np.int64).ravel()
        target_np = target.cpu().numpy().astype(np.int64).ravel()

        valid     = target_np != self.ignore_index
        pred_np   = pred_np[valid]
        target_np = target_np[valid]

        np.add.at(self.confusion, (target_np, pred_np), 1)

    def compute(self) -> dict:
        """
        Returns a flat dict of all metrics.
        Keys: pixel_acc, mean_iou_all, mean_iou_fg,
              iou_<name>, precision_<name>, recall_<name>, f1_<name>
              for each class name.
        """
        conf = self.confusion.astype(np.float64)
        tp   = np.diag(conf)
        fp   = conf.sum(axis=0) - tp   # predicted as class c but aren't
        fn   = conf.sum(axis=1) - tp   # are class c but predicted otherwise

        iou       = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)
        pixel_acc = tp.sum() / (conf.sum() + 1e-6)

        result = {
            "pixel_acc":      float(pixel_acc),
            "mean_iou_all":   float(iou.mean()),
            "mean_iou_fg":    float(iou[1:].mean()),   # exclude background
        }

        for i, name in enumerate(self.class_names):
            result[f"iou_{name}"]       = float(iou[i])
            result[f"precision_{name}"] = float(precision[i])
            result[f"recall_{name}"]    = float(recall[i])
            result[f"f1_{name}"]        = float(f1[i])

        return result

    def format_summary(self) -> str:
        """Human-readable multi-line summary (for logger)."""
        m    = self.compute()
        hdr  = f"  {'Class':<20} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
        sep  = "  " + "-" * 56
        rows = [hdr, sep]
        for name in self.class_names:
            rows.append(
                f"  {name:<20} "
                f"{m[f'iou_{name}']:>8.4f} "
                f"{m[f'precision_{name}']:>8.4f} "
                f"{m[f'recall_{name}']:>8.4f} "
                f"{m[f'f1_{name}']:>8.4f}"
            )
        rows.append(sep)
        rows.append(
            f"  {'mIoU (all)':<20} {m['mean_iou_all']:>8.4f}   "
            f"{'mIoU (fg)':<15} {m['mean_iou_fg']:>8.4f}"
        )
        rows.append(f"  {'Pixel Accuracy':<20} {m['pixel_acc']:>8.4f}")
        return "\n".join(rows)
