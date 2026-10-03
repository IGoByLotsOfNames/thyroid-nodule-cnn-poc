"""Offline evaluation walkthrough; --cnn additionally fits a synthetic toy model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _imports():
    # Use this checkout, including from another cwd or without an installed package.
    for path in (ROOT, ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_examples():
    _imports()
    from thyroid_poc.data import safe_path

    examples = ROOT / "examples"
    expected = _json(examples / "demo-checksums.json")
    if set(expected) != {"demo-input.json", "recorded-measurement.json"}:
        raise ValueError("Demo checksum inventory is incomplete")
    for name, checksum in expected.items():
        if _sha(examples / name) != checksum:
            raise ValueError(f"Bundled example changed: {name}")
    fixture = _json(examples / "demo-input.json")
    if fixture.get("kind") != "hand_authored_synthetic_predictions":
        raise ValueError("This walkthrough requires the explicitly synthetic fixture")
    classes = fixture["class_names"]
    if len(classes) != 2 or len(set(classes)) != 2:
        raise ValueError("Expected two ordered synthetic class names")
    identities, groups, hashes = set(), {}, set()
    for sample in fixture["samples"]:
        source, group, split, label = (sample[key] for key in ("source", "group", "split", "label"))
        if source in identities or type(label) is not int or label not in (0, 1):
            raise ValueError("Synthetic sample identity or label is invalid")
        if not isinstance(group, str) or not group or split not in ("train", "validation", "test"):
            raise ValueError("Synthetic group/split is invalid")
        if group in groups and groups[group] != split:
            raise ValueError("Synthetic group crosses partitions")
        path = safe_path(examples / "images", source)
        if _sha(path) != sample["sha256"] or sample["sha256"] in hashes:
            raise ValueError("Synthetic image changed or was duplicated")
        if split != "train":
            p = sample.get("probability")
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Synthetic probability is invalid")
        identities.add(source)
        groups[group] = split
        hashes.add(sample["sha256"])
    image_root = examples / "images"
    entries = list(image_root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Synthetic image inventory cannot contain symlinks")
    actual = {path.relative_to(image_root).as_posix() for path in entries if path.is_file()}
    if actual != identities:
        raise ValueError("Synthetic image inventory differs from the verified fixture")
    for split in ("train", "validation", "test"):
        if {r["label"] for r in fixture["samples"] if r["split"] == split} != {0, 1}:
            raise ValueError("Each synthetic partition must contain both classes")
    recorded = _json(examples / "recorded-measurement.json")
    return fixture, recorded, expected["demo-input.json"]


def synthetic_walkthrough(fixture, fixture_hash):
    _imports()
    from thyroid_poc.metrics import binary_metrics
    from thyroid_poc.selection import choose_threshold

    validation = [r for r in fixture["samples"] if r["split"] == "validation"]
    selected = choose_threshold(
        [r["label"] for r in validation], [r["probability"] for r in validation]
    )
    test = [r for r in fixture["samples"] if r["split"] == "test"]
    labels, probabilities = [r["label"] for r in test], [r["probability"] for r in test]
    return dict(
        notice="Constructed synthetic images and hand-authored probabilities demonstrate evaluation mechanics; no CNN inference or medical accuracy is claimed.",
        class_names=fixture["class_names"],
        selection=selected,
        selected_metrics=binary_metrics(labels, probabilities, selected["threshold"]),
        fixed_metrics=binary_metrics(labels, probabilities, 0.5),
        samples=test,
        counts={
            split: sum(r["split"] == split for r in fixture["samples"])
            for split in ("train", "validation", "test")
        },
        fixture_sha256=fixture_hash,
    )


def cnn_smoke(output):
    # Environment controls must precede optional NumPy/TensorFlow imports.
    os.environ.update(
        CUDA_VISIBLE_DEVICES="-1",
        TF_ENABLE_ONEDNN_OPTS="0",
        TF_DETERMINISTIC_OPS="1",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    import tensorflow as tf

    from thyroid_poc.data import create_split
    from thyroid_poc.evaluate import evaluate_models, write_report
    from thyroid_poc.selection import select_models
    from thyroid_poc.train import train_model

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.get_visible_devices("GPU"):
        raise ValueError("Synthetic smoke mode requires CPU-only execution")
    folder = output / "cnn-smoke"
    manifest = create_split(
        ROOT / "examples/images",
        folder / "split",
        train=0.5,
        validation=0.25,
        seed=17,
        independent_images=True,
    )
    model = folder / "model.keras"
    train_model(folder / "split", model, epochs=1, batch_size=4, seed=17, pretrained=False)
    tf.keras.backend.clear_session()
    selection = select_models([model], folder / "split", batch_size=4)
    write_report(selection, folder / "selection.json")
    report = evaluate_models(
        [model], folder / "split", batch_size=4, selection=folder / "selection.json"
    )
    write_report(report, folder / "evaluation.json")
    return dict(
        notice="Actual CNN training, save/reload, validation selection and held-out prediction on generated colour images only. This is a software smoke test, not a medical experiment.",
        metrics=report["metrics"]["model_1"],
        class_names=manifest["class_names"],
        seed=17,
        epochs=1,
        threshold=report["metrics"]["model_1"]["threshold"],
    )


def run_demo(output=None, *, cnn=False):
    fixture, recorded, fixture_hash = load_examples()
    if cnn:
        missing = [
            name
            for name in ("tensorflow", "numpy", "PIL")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise ValueError(
                "CNN mode needs the optional runtime ("
                + ", ".join(missing)
                + "). See docs/demo.md; the default demo needs only Python."
            )
    synthetic = synthetic_walkthrough(fixture, fixture_hash)
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "demo-output" / ("run-" + stamp)
    output = Path(output).resolve()
    for protected in (ROOT / "examples", ROOT / "src", ROOT / "demo_support"):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Choose an output outside the source and bundled examples")
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(
        schema_version=1,
        kind="thyroid_demo",
        synthetic=synthetic,
        recorded_measurement=recorded,
        cnn_smoke=None,
    )
    if cnn:
        payload["cnn_smoke"] = cnn_smoke(output)
    from demo_support.report import render_report

    images = {
        row["source"]: "data:image/png;base64,"
        + base64.b64encode((ROOT / "examples/images" / row["source"]).read_bytes()).decode("ascii")
        for row in synthetic["samples"]
    }
    (output / "report.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "report.html").write_text(render_report(payload, images), encoding="utf-8")
    return output, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh output directory; default creates a unique demo-output/run-* folder",
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        help="Also run one CPU epoch on included synthetic PNGs; requires optional TensorFlow runtime",
    )
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or later is required")
    try:
        output, payload = run_demo(args.output, cnn=args.cnn)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Demo stopped: {error}\n")
    synthetic = payload["synthetic"]
    print("Thyroid CNN demonstration - synthetic walkthrough + recorded research")
    print("Synthetic probabilities are hand-authored, not CNN predictions.")
    print(f"Synthetic validation-selected threshold: {synthetic['selection']['threshold']:.3f}")
    print(
        f"Synthetic test balanced accuracy: selected={synthetic['selected_metrics']['balanced_accuracy']:.2%}; fixed 0.5={synthetic['fixed_metrics']['balanced_accuracy']:.2%}"
    )
    print(
        f"Recorded TIRADS mean balanced accuracy: {payload['recorded_measurement']['across_seeds']['balanced_accuracy']['mean']:.2%} (five seeds; no rerun)"
    )
    if args.cnn:
        print("Synthetic CNN smoke: completed one epoch; see the separate report section.")
    print(f"Report: {output / 'report.html'}")
    print(f"Machine-readable output: {output / 'report.json'}")


if __name__ == "__main__":
    main()
