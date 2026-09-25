from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import image_datasets
from .models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a thyroid-nodule binary classifier")
    parser.add_argument("data", type=Path, help="Directory containing train/validation/test")
    parser.add_argument("--architecture", choices=("alexnet", "inception_v3", "inception_resnet_v2"), default="inception_v3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.keras"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    train, validation, _ = image_datasets(args.data, (224, 224), args.batch_size, args.seed)
    model = build_model(args.architecture)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.output, save_best_only=True),
    ]
    history = model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks)
    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

