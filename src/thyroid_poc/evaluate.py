"""Held-out evaluation with explicit fixed or validation-selected thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import load_metadata, metadata_path, validate_model
from .data import dataset, digest, manifest_fingerprint, validate_manifest
from .metrics import evaluation_report, validate_threshold


def _prepare_inputs(model_paths, root, *, batch_size):
    """Check the complete artifact set before importing TensorFlow or predicting."""
    paths = [Path(path) for path in model_paths]
    if not paths:
        raise ValueError("Provide at least one model")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Duplicate model path; every ensemble member must be distinct")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    root = Path(root)
    manifest = validate_manifest(root)
    fingerprint = manifest_fingerprint(manifest)
    test_hashes = {row["sha256"] for row in manifest["rows"] if row["split"] == "test"}
    records, seen_hashes = [], set()
    for path in paths:
        metadata = load_metadata(path)
        model_hash = metadata["model_sha256"]
        if model_hash in seen_hashes:
            raise ValueError("Duplicate model bytes; every ensemble member must be distinct")
        seen_hashes.add(model_hash)
        if metadata["class_names"] != manifest["class_names"]:
            raise ValueError("Model class order does not match the dataset")
        if set(metadata["trained_split_hashes"]) & test_hashes:
            raise ValueError("Test sample overlaps model training/validation history")
        if metadata["split_manifest_fingerprint"] != fingerprint:
            raise ValueError(
                "Use the model's recorded dataset manifest; external validation needs a separate protocol"
            )
        if metadata["source_fingerprint"] != manifest["source_fingerprint"]:
            raise ValueError("Model source fingerprint does not match the dataset")
        identity = dict(
            file=path.name,
            sha256=model_hash,
            metadata_sha256=digest(metadata_path(path)),
            architecture=metadata["architecture"],
        )
        records.append((path, metadata, identity))
    return root, manifest, records


def _predict_models(records, root, manifest, split, *, batch_size):
    """Internal adapter: selection calls validation; final evaluation calls test."""
    if split not in ("validation", "test"):
        raise ValueError("Predictions require validation or test split")
    import numpy as np
    import tensorflow as tf

    rows = [row for row in manifest["rows"] if row["split"] == split]
    predictions, identities = {}, {}
    for index, (path, metadata, identity) in enumerate(records):
        if (
            digest(path) != identity["sha256"]
            or digest(metadata_path(path)) != identity["metadata_sha256"]
        ):
            raise ValueError("Model or metadata changed after validation")
        model = tf.keras.models.load_model(path, compile=False)
        validate_model(model, metadata)
        ds = dataset(root, manifest, split, tuple(metadata["image_size"]), batch_size)
        probability = np.concatenate(
            [np.asarray(model(x, training=False)).reshape(-1) for x, _ in ds]
        )
        name = f"model_{index + 1}"
        predictions[name] = probability
        identities[name] = identity
    return rows, predictions, identities


def evaluate_models(model_paths, root, *, batch_size=32, threshold=None, selection=None):
    """Evaluate test data with a fixed threshold (default .5) or frozen selection."""
    if selection is not None and threshold is not None:
        raise ValueError("Use a fixed threshold OR a validation selection artifact")
    fixed_threshold = validate_threshold(0.5 if threshold is None else threshold)
    root, manifest, records = _prepare_inputs(model_paths, root, batch_size=batch_size)
    thresholds = None
    if selection is None:
        provenance = dict(
            mode="fixed",
            source="default" if threshold is None else "caller",
            threshold=fixed_threshold,
            selection_verified=False,
        )
    else:
        from .selection import load_selection

        artifact, artifact_hash = load_selection(selection, manifest, records)
        thresholds = {
            f"model_{index + 1}": artifact["decisions"][identity["sha256"]]["threshold"]
            for index, (_, _, identity) in enumerate(records)
        }
        if len(records) > 1:
            thresholds["ensemble"] = artifact["decisions"]["ensemble"]["threshold"]
        provenance = dict(
            mode="validation_selection",
            selection_verified=True,
            artifact_sha256=artifact_hash,
            objective=artifact["objective"],
            tie_break=artifact["tie_break"],
            candidate_policy=artifact["candidate_policy"],
            split="validation",
        )
    rows, predictions, identities = _predict_models(
        records, root, manifest, "test", batch_size=batch_size
    )
    report = evaluation_report(
        rows,
        predictions,
        manifest["class_names"],
        threshold=fixed_threshold,
        thresholds=thresholds,
        threshold_provenance=provenance,
    )
    report.update(
        models=identities,
        source_fingerprint=manifest["source_fingerprint"],
        manifest_sha256=digest(root / "manifest.json"),
        split="test",
    )
    return report


def write_report(report, output):
    output = Path(output)
    # Serialize before opening: invalid JSON must not leave a partial artifact.
    payload = json.dumps(report, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def add_threshold_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--threshold",
        type=float,
        help="Caller-fixed threshold; default .5, without selection evidence",
    )
    group.add_argument(
        "--selection", type=Path, help="Frozen validation-only threshold selection JSON"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on its validated held-out manifest"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    add_threshold_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    report = evaluate_models(
        [args.model],
        args.data,
        batch_size=args.batch_size,
        threshold=args.threshold,
        selection=args.selection,
    )
    write_report(report, args.output)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
