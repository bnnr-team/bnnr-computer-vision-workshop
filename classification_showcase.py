"""Workshop classification showcase for BNNR.

This script is beginner-friendly:
- no CLI parser,
- plain top-level constants,
- detailed comments,
- a compact model and a tiny dataset subset,
- live dashboard startup included in the script itself.

The companion multi-label script in the same folder is intentionally almost
identical so the two scripts are easily compared directly.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from bnnr import BNNRConfig, BNNRTrainer, SimpleTorchAdapter, start_dashboard
from bnnr.presets import auto_select_augmentations


# ---------------------------------------------------------------------------
# Workshop knobs
# ---------------------------------------------------------------------------
# These constants are meant to be edited
WITH_DASHBOARD = True
DASHBOARD_PORT = 8080
DASHBOARD_AUTO_OPEN = True

BATCH_SIZE = 32
TRAIN_SAMPLES = 256
VAL_SAMPLES = 128

M_EPOCHS = 5
MAX_ITERATIONS = 3
SEED = 42

REPORT_DIR = Path("reports") / "workshop_showcases" / "classification" # YOU CAN CNAGE THIS TO YOUR OWN PATH IF YOU WANT TO KEEP REPORTS FROM MULTIPLE RUNS
CHECKPOINT_DIR = Path("checkpoints") / "workshop_showcases" / "classification" # YOU CAN CNAGE THIS TO YOUR OWN PATH IF YOU WANT TO KEEP CHECKPOINTS FROM MULTIPLE RUNS

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# CIFAR-10 is a classic beginner dataset:
# - small RGB images (32x32),
# - 10 classes,
# - exactly one class label per image.
#
# We use only a small subset to keep the workshop demo quick on CPU.

# Beware: we do not normalize data for BNNR. That's because BNNR defines the augmentations itself
# Compose the transform with only ToTensor, which converts PIL images to tensors and scales pixel values to [0, 1].
transform = transforms.Compose([transforms.ToTensor()])

train_dataset_full = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)
val_dataset_full = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

train_dataset = Subset(train_dataset_full, range(TRAIN_SAMPLES))
val_dataset = Subset(val_dataset_full, range(VAL_SAMPLES))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# The network returns raw logits.
# For standard single-label classification we do NOT add softmax to the model:
# CrossEntropyLoss expects raw logits and applies the appropriate math itself.
class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), # modifying the input tensor in-place can save some memory, but be careful with it as it can lead to unexpected bugs if you reuse the same tensor elsewhere
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                           # -> [batch_size, 128, 1, 1]
        x = x.flatten(1)                               # -> [batch_size, 128]
        return self.classifier(x)                      # -> [batch_size, num_classes]


model = TinyClassifier(num_classes=10)


# ---------------------------------------------------------------------------
# Loss and optimizer
# ---------------------------------------------------------------------------
# This is the standard single-label recipe:
# - integer class ids,
# - CrossEntropyLoss,
# - one predicted class per image.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #parameters = weights and biases of the model


# ---------------------------------------------------------------------------
# BNNR configuration
# ---------------------------------------------------------------------------
# These metrics are intentionally small and readable
config = BNNRConfig(
    m_epochs=M_EPOCHS,
    max_iterations=MAX_ITERATIONS,
    selection_metric="f1_macro",
    metrics=["accuracy", "f1_macro", "loss"],
    seed=SEED,
    device="cpu",
    report_dir=REPORT_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
    event_log_enabled=True,
)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
# We start the dashboard explicitly before training so it can observe the run
# from the beginning and show live progress.
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
# This is the "manual but readable" API version of the BNNR flow:
# 1. wrap the PyTorch model in SimpleTorchAdapter,
# 2. let BNNR choose candidate augmentations,
# 3. construct BNNRTrainer,
# 4. run the iterative search.
adapter = SimpleTorchAdapter(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=config.device,
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
