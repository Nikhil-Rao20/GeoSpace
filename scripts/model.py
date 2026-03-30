"""
model.py — SegFormer-B4 wrapper for SVAMITVA segmentation.

Architecture:
    Encoder : MiT-B4 (Mix Transformer), pretrained on ImageNet-22K
              64M parameters, hierarchical 4-scale feature maps
    Decoder : All-MLP decode head (lightweight, 4-scale feature fusion)
    Output  : Bilinearly upsampled to input resolution (512 × 512)

Reference:
    Xie et al., "SegFormer: Simple and Efficient Design for Semantic
    Segmentation with Transformers", NeurIPS 2021.
    HuggingFace model: nvidia/mit-b4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


_CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Waterbody_poly",
    "Waterbody_line",
    "Utility_poly",
    "Bridge",
]


class SVAMITVASegFormer(nn.Module):
    """
    Thin wrapper around HuggingFace SegformerForSemanticSegmentation.

    Changes vs. raw HF model:
      - forward() returns logits at the *input* resolution (H, W) via
        bilinear upsampling from the native H/4 decoder output.
      - freeze_encoder() / unfreeze_all() helpers for cRT.
      - Human-readable param counts.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained:  str = "nvidia/mit-b4",
    ):
        super().__init__()
        self.num_classes = num_classes

        id2label = {i: _CLASS_NAMES[i] for i in range(num_classes)}
        label2id = {v: k for k, v in id2label.items()}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels        = num_classes,
            id2label          = id2label,
            label2id          = label2id,
            ignore_mismatched_sizes = True,   # re-init decode head for 7 classes
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  normalised float32 RGB tensor

        Returns:
            logits: (B, num_classes, H, W)  at input resolution
        """
        out    = self.model(pixel_values=x)
        logits = out.logits          # (B, C, H/4, W/4)  — native decoder output
        logits = F.interpolate(
            logits,
            size          = x.shape[-2:],
            mode          = "bilinear",
            align_corners = False,
        )
        return logits                # (B, C, H, W)

    # ── cRT helpers ───────────────────────────────────────────────────────────

    def freeze_encoder(self):
        """
        Freeze the MiT-B4 encoder (all SegformerEncoder parameters).
        Only the All-MLP decode head remains trainable — used for cRT.
        """
        for param in self.model.segformer.parameters():
            param.requires_grad = False
        n_frozen    = sum(p.numel() for p in self.model.segformer.parameters())
        n_trainable = self.get_trainable_params()
        print(
            f"[Model] Encoder frozen: {n_frozen/1e6:.1f} M params  |  "
            f"Trainable (decode head): {n_trainable/1e6:.1f} M params"
        )

    def unfreeze_all(self):
        """Unfreeze every parameter."""
        for param in self.parameters():
            param.requires_grad = True
        print(f"[Model] All {self.get_total_params()/1e6:.1f} M params unfrozen.")

    # ── Param counts ──────────────────────────────────────────────────────────

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_param_summary(self):
        total     = self.get_total_params()
        trainable = self.get_trainable_params()
        print(
            f"[Model] {self.model.config._name_or_path}  |  "
            f"Total: {total/1e6:.1f} M  |  "
            f"Trainable: {trainable/1e6:.1f} M  |  "
            f"Frozen: {(total-trainable)/1e6:.1f} M"
        )
