"""Freeze validation-only thresholds before held-out test evaluation.

Selection maximizes image-level balanced accuracy. This artifact records software
provenance, not a signature or proof of correct labels, patient IDs or data rights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import group_support, manifest_fingerprint
from .metrics import binary_metrics, prediction_columns, validate_probabilities

OBJECTIVE = "balanced_accuracy"
TIE_BREAK = "closest candidate to 0.5, then smaller candidate"
CANDIDATE_POLICY = (
    "unique validation probabilities plus 0, 0.5 and 1; positive when probability >= threshold"
)


def choose_threshold(labels, probabilities):
    """Find the optimal candidate in O(n log n); compare objectives as integers."""
    labels = list(labels)
    probabilities = validate_probabilities(probabilities)
    baseline = binary_metrics(labels, probabilities)
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Threshold selection requires both validation classes")
    candidates = sorted({0.0, 0.5, 1.0, *probabilities})
    ordered = sorted(zip(probabilities, labels))
    position, true_positive, true_negative = 0, positives, 0
    best_key, best_threshold, best_numerator = None, None, None
    for threshold in candidates:
        while position < len(ordered) and ordered[position][0] < threshold:
            label = ordered[position][1]
            true_positive -= label
            true_negative += 1 - label
            position += 1
        numerator = true_positive * negatives + true_negative * positives
        key = (-numerator, abs(threshold - 0.5), threshold)
        if best_key is None or key < best_key:
            best_key, best_threshold, best_numerator = key, threshold, numerator
    return dict(
        threshold=best_threshold,
        balanced_accuracy=best_numerator / (2 * positives * negatives),
        candidate_count=len(candidates),
        sample_count=baseline["sample_count"],
    )


def _model_bindings(records):
    return sorted(
        (
            dict(sha256=identity["sha256"], metadata_sha256=identity["metadata_sha256"])
            for _, _, identity in records
        ),
        key=lambda item: item["sha256"],
    )


def _selection_record(manifest, probabilities_by_hash, bindings):
    rows = [row for row in manifest["rows"] if row["split"] == "validation"]
    model_hashes = [item["sha256"] for item in bindings]
    if len(set(model_hashes)) != len(model_hashes) or not model_hashes:
        raise ValueError("Selection needs distinct model artifacts")
    if not isinstance(probabilities_by_hash, dict) or set(probabilities_by_hash) != set(
        model_hashes
    ):
        raise ValueError("Selection probabilities must match the exact model artifact set")
    probabilities = {
        key: validate_probabilities(probabilities_by_hash[key]) for key in sorted(model_hashes)
    }
    columns = prediction_columns(probabilities, len(rows))
    labels = [row["label"] for row in rows]
    return dict(
        schema_version=1,
        kind="validation_threshold_selection",
        split="validation",
        objective=OBJECTIVE,
        tie_break=TIE_BREAK,
        candidate_policy=CANDIDATE_POLICY,
        split_manifest_fingerprint=manifest_fingerprint(manifest),
        class_names=manifest["class_names"],
        model_bindings=bindings,
        validation_sample_count=len(rows),
        validation_class_counts=[labels.count(0), labels.count(1)],
        support=group_support(rows),
        validation_probabilities=probabilities,
        decisions={key: choose_threshold(labels, values) for key, values in columns.items()},
    )


def select_models(model_paths, root, *, batch_size=32):
    """Predict validation rows only and return an immutable-ready selection record."""
    from .evaluate import _predict_models, _prepare_inputs

    root, manifest, records = _prepare_inputs(model_paths, root, batch_size=batch_size)
    _, predictions, identities = _predict_models(
        records, root, manifest, "validation", batch_size=batch_size
    )
    by_hash = {identities[name]["sha256"]: values for name, values in predictions.items()}
    return _selection_record(manifest, by_hash, _model_bindings(records))


def load_selection(path, manifest, records):
    """Check exact bindings and recompute thresholds from saved validation evidence."""
    payload = Path(path).read_bytes()
    try:
        artifact = json.loads(payload)
        if not isinstance(artifact, dict) or type(artifact.get("schema_version")) is not int:
            raise ValueError("Invalid selection schema")
        expected = _selection_record(
            manifest, artifact["validation_probabilities"], _model_bindings(records)
        )
        if artifact != expected:
            raise ValueError(
                "Selection artifact does not match its model/metadata/manifest bindings or validation decisions"
            )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed validation selection artifact") from error
    return artifact, hashlib.sha256(payload).hexdigest()


def main():
    from .evaluate import write_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation-selection.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh selection output path")
    artifact = select_models(args.models, args.data, batch_size=args.batch_size)
    write_report(artifact, args.output)
    print(json.dumps(artifact["decisions"], indent=2))


if __name__ == "__main__":
    main()
