from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .data import image_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained thyroid-nodule classifier")
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()

    _, _, test = image_datasets(args.data, (224, 224), args.batch_size, seed=42)
    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(test, verbose=0).reshape(-1)
    labels = np.concatenate([batch_labels.numpy().reshape(-1) for _, batch_labels in test])
    predictions = (probabilities >= 0.5).astype(int)
    report = {
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(labels, predictions, output_dict=True),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "threshold": 0.5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

