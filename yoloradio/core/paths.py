from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path.cwd().resolve()
DATASETS_DIR = PROJECT_DIR / "Datasets"
MODELS_DIR = PROJECT_DIR / "Models"
LOGS_DIR = PROJECT_DIR / "runs"
MODELS_PRETRAINED_DIR = MODELS_DIR / "pretrained"
MODELS_TRAINED_DIR = MODELS_DIR / "trained"

# Ensure basic directories
for p in (DATASETS_DIR, MODELS_DIR, MODELS_PRETRAINED_DIR, MODELS_TRAINED_DIR):
    p.mkdir(parents=True, exist_ok=True)
