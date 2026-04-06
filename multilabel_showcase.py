"""Workshop multi-label classification showcase for BNNR.

This script mirrors ``classification_showcase.py`` on purpose.
The main teaching goal is to let compare the two files and notice
that only a few important lines really change for multi-label learning.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bnnr import BNNRConfig, BNNRTrainer, SimpleTorchAdapter, start_dashboard
from bnnr.presets import auto_select_augmentations


# ---------------------------------------------------------------------------
# Workshop knobs
# ---------------------------------------------------------------------------
WITH_DASHBOARD = True
DASHBOARD_PORT = 8080
DASHBOARD_AUTO_OPEN = True

BATCH_SIZE = 32
TRAIN_SAMPLES = 256
VAL_SAMPLES = 128

M_EPOCHS = 5
MAX_ITERATIONS = 3
SEED = 42

NUM_LABELS = 6
IMAGE_SIZE = 32

REPORT_DIR = Path("reports") / "workshop_showcases" / "multilabel"
CHECKPOINT_DIR = Path("checkpoints") / "workshop_showcases" / "multilabel"

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# This synthetic dataset is "VOC-style" in the teaching sense:
# one image can contain several active labels at the same time.
#
# Each sample returns:
# - an RGB image tensor,
# - a multi-hot label vector of shape [NUM_LABELS].
class SyntheticVOCStyleDataset(Dataset):
    def __init__(self, size: int, *, seed: int) -> None:
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index) # deterministic randomness per sample

        image = torch.rand((3, IMAGE_SIZE, IMAGE_SIZE), generator=generator) * 0.10 # mostly dark background
        label = torch.zeros(NUM_LABELS, dtype=torch.float32) # multi-hot vector of labels, initialized to all zeros

        # In multi-label classification, several labels can be active together.
        active_count = int(torch.randint(1, 4, (1,), generator=generator).item()) # randomly choose how many labels are active for this sample (between 1 and 3)
        active_labels = torch.randperm(NUM_LABELS, generator=generator)[:active_count]
        label[active_labels] = 1.0

        # We draw a few simple visual patterns so the task stays intuitive.
        for active in active_labels.tolist():
            if active == 0:
                image[0] += 0.45  # red tint
            elif active == 1:
                image[1] += 0.45  # green tint
            elif active == 2:
                image[2] += 0.45  # blue tint
            elif active == 3:
                image[:, ::4, :] += 0.30  # horizontal stripes
            elif active == 4:
                image[:, :, ::4] += 0.30  # vertical stripes
            elif active == 5:
                image[:, 10:22, 10:22] += 0.40  # bright central patch

        image = image.clamp(0.0, 1.0)
        return image, label


train_dataset = SyntheticVOCStyleDataset(TRAIN_SAMPLES, seed=SEED)
val_dataset = SyntheticVOCStyleDataset(VAL_SAMPLES, seed=SEED + 1)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# The model still returns raw logits, but now there is one logit per label.
# We still avoid adding sigmoid to the model, because BCEWithLogitsLoss expects
# raw logits and applies the sigmoid internally in a numerically stable way.
class TinyMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels: int = NUM_LABELS) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


model = TinyMultiLabelClassifier(num_labels=NUM_LABELS)


# ---------------------------------------------------------------------------
# Loss and optimizer
# ---------------------------------------------------------------------------
# These are the key deep-learning changes versus the classification script:
# - targets are multi-hot vectors,
# - the model predicts one logit per label,
# - we use BCEWithLogitsLoss.
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# BNNR configuration
# ---------------------------------------------------------------------------
# These are the key BNNR changes versus the classification script:
# - task="multilabel"
# - f1_samples as the main selection metric
config = BNNRConfig(
    task="multilabel",
    m_epochs=M_EPOCHS,
    max_iterations=MAX_ITERATIONS,
    selection_metric="f1_samples",
    metrics=["f1_samples", "f1_macro", "accuracy", "loss"],
    seed=SEED,
    device="cpu",
    report_dir=REPORT_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    event_log_enabled=True,
)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
if WITH_DASHBOARD:
    dashboard_url = start_dashboard(
        config.report_dir,
        port=DASHBOARD_PORT,
        auto_open=DASHBOARD_AUTO_OPEN,
    )
    print(f"Dashboard: {dashboard_url}")


# ---------------------------------------------------------------------------
# Adapter and trainer
# ---------------------------------------------------------------------------
# This block is almost identical to the classification script.
# The intended workshop lesson is that the main BNNR workflow stays the same.
adapter = SimpleTorchAdapter(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=config.device,
    multilabel=True,
)
augmentations = auto_select_augmentations(random_state=config.seed)
trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, config)
result = trainer.run()


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------
m = result.config.selection_metric
print(f"Best {m}: {result.best_metrics.get(m, float('nan')):.4f}")
print(f"Report: {result.report_json_path}")
