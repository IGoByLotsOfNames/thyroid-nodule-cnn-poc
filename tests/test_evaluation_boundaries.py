"""Pre-inference experiment boundaries and real command-line failure paths.

Generated images and inert model bytes are sufficient: every negative path must
reject the request before a model can be deserialized or used for prediction.
"""

import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from thyroid_poc import evaluate
from thyroid_poc.artifacts import metadata_path, save_metadata
from thyroid_poc.data import create_split
from thyroid_poc.selection import _model_bindings, _selection_record, load_selection, select_models


class EvaluationBoundaryFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for label, name in enumerate(("synthetic_a", "synthetic_b")):
            (self.root / "source" / name).mkdir(parents=True)
            for index in range(6):
                Image.new("RGB", (4, 4), (label * 180, index * 20, 30)).save(
                    self.root / "source" / name / f"{index}.png"
                )
        self.data = self.root / "split"
        self.manifest = create_split(self.root / "source", self.data, independent_images=True)
        self.models = [self.root / "a.keras", self.root / "b.keras"]
        self.metadata = []
        for index, path in enumerate(self.models):
            path.write_bytes(f"inert metadata-boundary model {index}".encode())
            self.metadata.append(
                save_metadata(path, self.manifest, size=(4, 4), architecture="fixture", seed=42)
            )

    def put_metadata(self, metadata, index=0):
        metadata_path(self.models[index]).write_text(json.dumps(metadata), encoding="utf-8")

    def prepared(self):
        return evaluate._prepare_inputs(self.models[:1], self.data, batch_size=2)

    def selection_record(self):
        _, manifest, records = self.prepared()
        values = [
            0.2 if row["label"] == 0 else 0.8
            for row in manifest["rows"]
            if row["split"] == "validation"
        ]
        by_hash = {records[0][2]["sha256"]: values}
        return _selection_record(manifest, by_hash, _model_bindings(records))

    def cli(self, entry, arguments):
        # Preserve coverage subprocess variables and every other inherited setting.
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, "-B", "-m", f"thyroid_poc.{entry}", *map(str, arguments)],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )


class MetadataAndPredictionBoundaryTests(EvaluationBoundaryFixture):
    @classmethod
    def setUpClass(cls):
        # sys.modules mock scopes restore their snapshot: preload this real
        # dependency so it is not removed and reimported after each fake backend.
        importlib.import_module("numpy")

    def test_incompatible_metadata_rejected_before_prediction(self):
        test_hash = next(row["sha256"] for row in self.manifest["rows"] if row["split"] == "test")
        cases = (
            ("class_names", self.manifest["class_names"][::-1], "class order"),
            ("source_fingerprint", "0" * 64, "source fingerprint"),
            ("split_manifest_fingerprint", "0" * 64, "recorded dataset manifest"),
            ("trained_split_hashes", [test_hash], "overlaps"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                metadata = copy.deepcopy(self.metadata[0])
                metadata[field] = value
                self.put_metadata(metadata)
                with (
                    patch.dict(sys.modules, {"tensorflow": None}),
                    patch.object(evaluate, "_predict_models") as predict,
                ):
                    for operation in (evaluate.evaluate_models, select_models):
                        with self.assertRaisesRegex(ValueError, message):
                            operation(self.models, self.data)
                    predict.assert_not_called()

    def test_training_history_requires_canonical_hash_list(self):
        held_out = next(row["sha256"] for row in self.manifest["rows"] if row["split"] == "test")
        malformed = (
            held_out,
            {held_out: True},
            None,
            [123],
            [""],
            ["a" * 63],
            ["g" * 64],
            [held_out.upper()],
        )
        for history in malformed:
            with self.subTest(history=history):
                metadata = copy.deepcopy(self.metadata[0])
                metadata["trained_split_hashes"] = history
                self.put_metadata(metadata)
                with (
                    patch.dict(sys.modules, {"tensorflow": None}),
                    patch.object(evaluate, "_predict_models") as predict,
                ):
                    with self.assertRaisesRegex(ValueError, "history"):
                        evaluate.evaluate_models(self.models, self.data)
                    predict.assert_not_called()
        # Duplicate source images can legitimately repeat a training hash.
        metadata = copy.deepcopy(self.metadata[0])
        metadata["trained_split_hashes"].append(metadata["trained_split_hashes"][0])
        self.put_metadata(metadata)
        self.assertEqual(1, len(self.prepared()[2]))

    def test_missing_training_history_is_rejected(self):
        metadata = copy.deepcopy(self.metadata[0])
        del metadata["trained_split_hashes"]
        self.put_metadata(metadata)
        with patch.object(evaluate, "_predict_models") as predict:
            with self.assertRaisesRegex(ValueError, "history"):
                evaluate.evaluate_models(self.models, self.data)
            predict.assert_not_called()

    def test_invalid_batch_sizes_fail_before_opening_dataset(self):
        with (
            patch.object(evaluate, "validate_manifest") as validate,
            patch.dict(sys.modules, {"tensorflow": None}),
        ):
            for batch_size in (0, -1, 1.5, True):
                with self.subTest(batch_size=batch_size):
                    for operation in (evaluate.evaluate_models, select_models):
                        with self.assertRaisesRegex(ValueError, "Batch size"):
                            operation(self.models, self.root / "absent", batch_size=batch_size)
            validate.assert_not_called()

    def test_changed_artifact_after_preparation_is_not_loaded(self):
        model = self.models[0]
        for changed_path in (model, metadata_path(model)):
            with self.subTest(path=changed_path.name):
                root, manifest, records = self.prepared()
                original = changed_path.read_bytes()
                changed_path.write_bytes(original + b"\n")
                loader = Mock()
                backend = SimpleNamespace(
                    keras=SimpleNamespace(models=SimpleNamespace(load_model=loader))
                )
                try:
                    with (
                        patch.dict(sys.modules, {"tensorflow": backend}),
                        patch.object(evaluate, "dataset") as datasets,
                    ):
                        with self.assertRaisesRegex(ValueError, "changed after validation"):
                            evaluate._predict_models(records, root, manifest, "test", batch_size=2)
                        loader.assert_not_called()
                        datasets.assert_not_called()
                finally:
                    changed_path.write_bytes(original)

    def test_incompatible_model_shapes_fail_before_dataset_or_inference(self):
        root, manifest, records = self.prepared()
        shapes = (
            ((None, 3, 4, 3), (None, 1)),
            ((None, 4, 4, 3), (None, 2)),
            ([(None, 4, 4, 3), (None, 4, 4, 3)], (None, 1)),
            ((None, 4, 4, 3), [(None, 1), (None, 1)]),
        )
        for input_shape, output_shape in shapes:
            with self.subTest(input_shape=input_shape, output_shape=output_shape):
                model = Mock(input_shape=input_shape, output_shape=output_shape)
                loader = Mock(return_value=model)
                backend = SimpleNamespace(
                    keras=SimpleNamespace(models=SimpleNamespace(load_model=loader))
                )
                with (
                    patch.dict(sys.modules, {"tensorflow": backend}),
                    patch.object(evaluate, "dataset") as datasets,
                ):
                    with self.assertRaisesRegex(ValueError, "input/output shape"):
                        evaluate._predict_models(records, root, manifest, "test", batch_size=2)
                    datasets.assert_not_called()
                    model.assert_not_called()

    def test_malformed_selection_json_and_evidence_fail_closed(self):
        _, manifest, records = self.prepared()
        valid = self.selection_record()
        bad_evidence = copy.deepcopy(valid)
        first_hash = next(iter(bad_evidence["validation_probabilities"]))
        bad_evidence["validation_probabilities"][first_hash][0] = float("nan")
        payloads = (
            "{",
            "null",
            "[]",
            "{}",
            json.dumps({**valid, "schema_version": True}),
            json.dumps({**valid, "validation_probabilities": None}),
            json.dumps(bad_evidence),
        )
        path = self.root / "selection.json"
        for payload in payloads:
            with self.subTest(payload=payload[:50]):
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_selection(path, manifest, records)


class EvaluationCliBoundaryTests(EvaluationBoundaryFixture):
    def test_cli_rejects_conflicting_threshold_options(self):
        output = self.root / "report.json"
        for entry, prefix in (
            ("evaluate", [self.models[0], self.data]),
            ("ensemble", [self.data, *self.models]),
        ):
            with self.subTest(entry=entry):
                result = self.cli(
                    entry,
                    [
                        *prefix,
                        "--threshold",
                        ".5",
                        "--selection",
                        self.root / "absent.json",
                        "--output",
                        output,
                    ],
                )
                self.assertEqual(2, result.returncode, result.stderr)
                self.assertIn("not allowed with argument", result.stderr)
                self.assertFalse(output.exists())

    def test_cli_model_cardinality_errors(self):
        for entry, arguments in (
            ("ensemble", [self.data, self.models[0]]),
            ("selection", [self.data]),
        ):
            with self.subTest(entry=entry):
                result = self.cli(entry, arguments)
                self.assertEqual(2, result.returncode, result.stderr)
                self.assertFalse((self.root / "artifacts").exists())

    def test_cli_exact_copies_rejected_by_ensemble_and_selection(self):
        duplicate = self.root / "copy.keras"
        shutil.copyfile(self.models[0], duplicate)
        shutil.copyfile(metadata_path(self.models[0]), metadata_path(duplicate))
        output = self.root / "report.json"
        for entry in ("ensemble", "selection"):
            with self.subTest(entry=entry):
                result = self.cli(entry, [self.data, self.models[0], duplicate, "--output", output])
                self.assertEqual(1, result.returncode, result.stderr)
                self.assertIn("Duplicate model bytes", result.stderr)
                self.assertFalse(output.exists())

    def test_cli_nonfinite_threshold_rejected_before_data_loading(self):
        output = self.root / "report.json"
        result = self.cli(
            "evaluate",
            [self.models[0], self.root / "absent", "--threshold", "nan", "--output", output],
        )
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn("Threshold must be a finite number", result.stderr)
        self.assertFalse(output.exists())

    def test_cli_selection_preserves_existing_output_before_loading_inputs(self):
        output = self.root / "existing.json"
        original = b'{"previous": "keep exactly these bytes"}'
        output.write_bytes(original)
        result = self.cli(
            "selection", [self.root / "absent-data", self.root / "absent.keras", "--output", output]
        )
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn("Choose a fresh selection output path", result.stderr)
        self.assertEqual(original, output.read_bytes())

    def test_cli_rejects_stale_metadata_binding_without_report(self):
        selection = self.root / "selection.json"
        selection.write_text(json.dumps(self.selection_record()), encoding="utf-8")
        metadata = copy.deepcopy(self.metadata[0])
        metadata["training_note"] = "changed since validation selection"
        self.put_metadata(metadata)
        output = self.root / "report.json"
        result = self.cli(
            "evaluate", [self.models[0], self.data, "--selection", selection, "--output", output]
        )
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertIn("Selection artifact does not match", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
