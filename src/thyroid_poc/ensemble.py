from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import image_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Average predictions from multiple Keras models")
    parser.add_argument("data", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    _, _, test = image_datasets(args.data, (224, 224), args.batch_size, seed=42)
    predictions = []
    for model_path in args.models:
        model = tf.keras.models.load_model(model_path)
        predictions.append(model.predict(test, verbose=0).reshape(-1))
    ensemble = np.mean(np.vstack(predictions), axis=0)
    print(f"Generated {len(ensemble)} ensemble probabilities from {len(predictions)} models")


if __name__ == "__main__":
    main()

