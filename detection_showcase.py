"""Workshop detection showcase for BNNR.

This script mirrors the existing classification and multilabel showcases on
purpose: plain top-level knobs, a tiny dataset, a small model, and a compact
BNNR training flow. The only twist is that detection is more expensive, so we
force a quick run configuration directly in the script.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bnnr import (
    BNNRConfig,
    BNNRTrainer,
    DetectionAdapter,
    DetectionHorizontalFlip,
    DetectionICD,
    DetectionRandomScale,
    detection_collate_fn_with_index,
    start_dashboard,
)


# ---------------------------------------------------------------------------
# Workshop knobs
# ---------------------------------------------------------------------------
# Detection demo knobs (CPU-friendly defaults for a quick run)
QUICK_RUN = True

WITH_DASHBOARD = True
DASHBOARD_PORT = 8080
DASHBOARD_AUTO_OPEN = True

# Model / data knobs (single lightweight pretrained backbone)
USE_PRETRAINED_BACKBONE = True  # set False to avoid downloading weights

BATCH_SIZE = 2
TRAIN_SAMPLES = 48
VAL_SAMPLES = 16

M_EPOCHS = 1 if QUICK_RUN else 5
MAX_ITERATIONS = 1 if QUICK_RUN else 3
SEED = 42

IMAGE_SIZE = 128
NUM_FOREGROUND_CLASSES = 3
DETECTION_CLASS_NAMES = ["background", "square", "circle", "diamond"]

REPORT_DIR = Path("reports") / "workshop_showcases" / "detection"
CHECKPOINT_DIR = Path("checkpoints") / "workshop_showcases" / "detection"

torch.manual_seed(SEED)

# CPU-friendly threading
torch.set_num_threads(min(4, (os.cpu_count() or 1)))

_CLASS_COLORS = [
    torch.tensor([0.90, 0.20, 0.20], dtype=torch.float32),
    torch.tensor([0.20, 0.90, 0.20], dtype=torch.float32),
    torch.tensor([0.20, 0.35, 0.95], dtype=torch.float32),
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# tiny procedural dataset with clean shapes on top of a low-contrast background.
class FakeDataDetectionDataset(Dataset):
    def __init__(self, size: int, *, seed: int, image_size: int = IMAGE_SIZE) -> None:
        self.size = size
        self.seed = seed
        self.image_size = image_size

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _draw_square(
        image: torch.Tensor,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        color: torch.Tensor,
    ) -> None:
        image[:, y1:y2, x1:x2] = color.view(3, 1, 1)

    @staticmethod
    def _draw_circle(
        image: torch.Tensor,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        color: torch.Tensor,
    ) -> None:
        height = max(1, y2 - y1)
        width = max(1, x2 - x1)
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        mask = ((yy - cy) / max(height / 2.0, 1.0)) ** 2 + ((xx - cx) / max(width / 2.0, 1.0)) ** 2 <= 1.0
        patch = image[:, y1:y2, x1:x2]
        patch[:, mask] = color.view(3, 1)

    @staticmethod
    def _draw_diamond(
        image: torch.Tensor,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
        color: torch.Tensor,
    ) -> None:
        height = max(1, y2 - y1)
        width = max(1, x2 - x1)
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        mask = (torch.abs(xx - cx) / max(width / 2.0, 1.0) + torch.abs(yy - cy) / max(height / 2.0, 1.0)) <= 1.0
        patch = image[:, y1:y2, x1:x2]
        patch[:, mask] = color.view(3, 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
        rng = random.Random(self.seed + index)
        noise_gen = torch.Generator().manual_seed(self.seed * 997 + index)
        image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)

        base_colors = [
            rng.uniform(0.06, 0.18),
            rng.uniform(0.06, 0.18),
            rng.uniform(0.06, 0.18),
        ]
        axis = 0 if rng.random() > 0.5 else 1
        for ch, base in enumerate(base_colors):
            grad = torch.linspace(base, base + rng.uniform(0.05, 0.12), self.image_size)
            image[ch] = grad.unsqueeze(axis).expand(self.image_size, self.image_size)
        image += torch.randn(3, self.image_size, self.image_size, generator=noise_gen) * 0.015
        image.clamp_(0.0, 1.0)

        n_objects = 1 + (index % 2)
        boxes: list[list[float]] = []
        labels: list[int] = []

        for obj_idx in range(n_objects):
            class_id = ((index + obj_idx) % NUM_FOREGROUND_CLASSES) + 1
            box_w = rng.randint(self.image_size // 5, self.image_size // 3)
            box_h = rng.randint(self.image_size // 5, self.image_size // 3)
            x1 = rng.randint(0, max(0, self.image_size - box_w - 1))
            y1 = rng.randint(0, max(0, self.image_size - box_h - 1))
            x2 = min(self.image_size, x1 + box_w)
            y2 = min(self.image_size, y1 + box_h)

            color = _CLASS_COLORS[class_id - 1]
            if class_id == 1:
                self._draw_square(image, y1, y2, x1, x2, color)
            elif class_id == 2:
                self._draw_circle(image, y1, y2, x1, x2, color)
            else:
                self._draw_diamond(image, y1, y2, x1, x2, color)

            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(class_id)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image.clamp(0.0, 1.0), target, index


def build_datasets() -> tuple[Dataset, Dataset]:
    train_dataset = FakeDataDetectionDataset(TRAIN_SAMPLES, seed=SEED, image_size=IMAGE_SIZE)
    val_dataset = FakeDataDetectionDataset(VAL_SAMPLES, seed=SEED + 1, image_size=IMAGE_SIZE)
    return train_dataset, val_dataset


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=detection_collate_fn_with_index, # is useful for some augmentations that need to know the sample index
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=detection_collate_fn_with_index, # stacks targets with different number of boxes, and also provides sample indices
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# We use a torchvision detection model 
class TinyDetectionModel(nn.Module):
    def __init__(self, num_classes: int = len(DETECTION_CLASS_NAMES)) -> None:
        super().__init__()
        # Use a single lightweight detection backbone: MobileNet v3 with FPN.
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        try:
            from torchvision.models import MobileNet_V3_Large_Weights
            weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if USE_PRETRAINED_BACKBONE else None
        except Exception:
            weights_backbone = None

        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            min_size=IMAGE_SIZE,
            max_size=IMAGE_SIZE,
        )

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        return self.model(images, targets)


def build_model() -> TinyDetectionModel:
    return TinyDetectionModel(num_classes=len(DETECTION_CLASS_NAMES))


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------
def build_detection_augmentations() -> list:
    return [
        DetectionHorizontalFlip(
            probability=0.5,
            name_override="det_hflip",
            random_state=SEED,
        ),
        DetectionRandomScale(
            probability=0.5,
            scale_range=(0.8, 1.2),
            name_override="det_scale",
            random_state=SEED + 1,
        ),
        DetectionICD(
            probability=0.3,
            name_override="det_icd",
            random_state=SEED + 2,
        ),
    ]


# ---------------------------------------------------------------------------
# BNNR configuration
# ---------------------------------------------------------------------------
def build_config() -> BNNRConfig:
    return BNNRConfig(
        task="detection",
        m_epochs=M_EPOCHS,
        max_iterations=MAX_ITERATIONS,
        selection_metric="map_50",
        metrics=["map_50", "map_50_95", "loss"],
        seed=SEED,
        device="cpu",
        report_dir=REPORT_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        event_log_enabled=True,
        detection_class_names=DETECTION_CLASS_NAMES,
        candidate_pruning_enabled=True,
        candidate_pruning_relative_threshold=0.7, 
        candidate_pruning_warmup_epochs=1,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def maybe_start_dashboard(config: BNNRConfig) -> str:
    if not WITH_DASHBOARD:
        return ""
    dashboard_url = start_dashboard(
        config.report_dir,
        port=DASHBOARD_PORT,
        auto_open=DASHBOARD_AUTO_OPEN,
    )
    print(f"Dashboard: {dashboard_url}")
    return dashboard_url


# ---------------------------------------------------------------------------
# Adapter and trainer
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Run the showcase (simple, top-level flow similar to other showcase scripts)
# ---------------------------------------------------------------------------
config = build_config()

# Start dashboard early so it captures run from the beginning
if WITH_DASHBOARD:
    dashboard_url = start_dashboard(config.report_dir, port=DASHBOARD_PORT, auto_open=DASHBOARD_AUTO_OPEN)
    print(f"Dashboard: {dashboard_url}")

train_loader, val_loader = build_dataloaders()
model = build_model()

# Lightweight optimizer for short CPU-friendly runs
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
adapter = DetectionAdapter(model=model, optimizer=optimizer, device=config.device)
augmentations = build_detection_augmentations()

print()
print("=" * 68)
print("  BNNR  ·  Detection Showcase")
print("-" * 68)
print(f"  Dataset                : Procedural synthetic shapes")
print(f"  Backbone               : mobilenet (pretrained={USE_PRETRAINED_BACKBONE})")
print(f"  Train / Val samples    : {TRAIN_SAMPLES} / {VAL_SAMPLES}")
print(f"  Batch size             : {BATCH_SIZE}")
print(f"  Max main-path epochs   : ~{config.m_epochs * (config.max_iterations + 1)}")
print(f"  Decision rounds        : {config.max_iterations}")
print(f"  Epochs per branch      : {config.m_epochs}")
print(f"  Augmentation candidates: {len(augmentations)}")
print(f"  Image size             : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Device                 : {config.device}")
print(f"  Mode                   : {'QUICK' if QUICK_RUN else 'FULL'}")
print("=" * 68)
print()

trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config)
result = trainer.run()

print()
print("=" * 68)
print("  Detection Showcase — Results")
print("-" * 68)
print(f"  Best path      : {result.best_path}")
print(f"  Best metrics   : {result.best_metrics}")
print(f"  Report JSON    : {result.report_json_path}")
print("=" * 68)
print()
