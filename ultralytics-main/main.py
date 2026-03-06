from __future__ import annotations

import os
from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
LAST_PT = WORKSPACE / "runs" / "detect" / "train" / "weights" / "last.pt"
DATA_YAML = ROOT / "datasets" / "VisDrone.yaml"


def main() -> None:
    if not LAST_PT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {LAST_PT}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {DATA_YAML}")

    # Windows avoids multiprocessing bootstrap errors with workers=0.
    workers = 0 if os.name == "nt" else 8

    model = YOLO(str(LAST_PT))
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=416,
        batch=16,
        workers=8,
        project=str(WORKSPACE / "runs" / "detect"),
        name="train_continue100",
    )


if __name__ == "__main__":
    freeze_support()
    main()
