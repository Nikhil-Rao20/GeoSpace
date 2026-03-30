"""
stage2_model.py — Lightweight EfficientNet classifiers for Stage-2 subtypes.

Building  → EfficientNet-B2 (9.1M params)  — larger because building ROIs
            are more visually complex (texture/material differences).
Road      → EfficientNet-B0 (5.3M params)
Waterbody → EfficientNet-B0 (5.3M params)
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
)


_BACKBONE_MAP = {
    "building":  ("b2", 1408),
    "road":      ("b0",  1280),
    "waterbody": ("b0",  1280),
}


class SubtypeClassifier(nn.Module):
    """
    EfficientNet backbone with a two-layer classification head.

    Head:
        GlobalAvgPool → Linear(features, 256) → GELU → Dropout(0.3)
                      → Linear(256, num_classes)

    Args:
        cls_name:    "building" | "road" | "waterbody"
        num_classes: number of output classes
        dropout:     dropout rate before final linear
    """

    def __init__(
        self,
        cls_name:    str,
        num_classes: int,
        dropout:     float = 0.3,
    ):
        super().__init__()
        variant, feat_dim = _BACKBONE_MAP[cls_name]
        self.cls_name    = cls_name
        self.num_classes = num_classes

        if variant == "b2":
            base = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Keep features (conv + BN + pool), replace classifier head
        self.features   = base.features
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)  normalised RGB
        returns: (B, num_classes)  logits
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)