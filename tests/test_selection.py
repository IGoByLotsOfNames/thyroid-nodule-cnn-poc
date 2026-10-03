"""Threshold provenance and ensemble identity regressions; no face/medical data."""

import copy
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from thyroid_poc import evaluate
from thyroid_poc.artifacts import metadata_path, save_metadata
from thyroid_poc.data import create_split, digest
from thyroid_poc.metrics import binary_metrics, evaluation_report
from thyroid_poc.selection import choose_threshold, load_selection, select_models


def image_fixture(root):
    for label, name in enumerate(("synthetic_a", "synthetic_b")):
        (root / name).mkdir(parents=True)
        for index in range(6):
            Image.new("RGB", (8, 8), (label * 180, index * 20, 40)).save(
                root / name / f"{index}.png"
            )


class ThresholdChoiceTests(unittest.TestCase):
    def test_nondefault_threshold_and_tie_rules(self):
        self.assertEqual(0.4, choose_threshold([0, 1], [0.2, 0.4])["threshold"])
        self.assertEqual(0.8, choose_threshold([0, 1], [0.6, 0.8])["threshold"])
        self.assertEqual(0.5, choose_threshold([0, 1], [0.5, 0.5])["threshold"])
        # .25 and .75 tie for best balanced accuracy; .5 is worse.
        result = choose_threshold([0, 1, 0, 1], [0, 0.25, 0.5, 0.75])
        self.assertEqual(0.25, result["threshold"])
        self.assertEqual(0.75, result["balanced_accuracy"])

    def test_randomized_exhaustive_threshold_oracle(self):
        rng = random.Random(83)
        for _ in range(60):
            labels = [0] * rng.randrange(1, 8) + [1] * rng.randrange(1, 8)
            scores = [rng.randrange(9) / 8 for _ in labels]
            candidates = set(scores) | {0.0, 0.5, 1.0}
            expected = min(
                candidates,
                key=lambda t: (
                    -binary_metrics(labels, scores, t)["balanced_accuracy"],
                    abs(t - 0.5),
                    t,
                ),
            )
            self.assertEqual(expected, choose_threshold(labels, scores)["threshold"])

    def test_invalid_or_single_class_inputs(self):
        for labels, scores in (
            ([1, 1], [0.2, 0.8]),
            ([], []),
            ([0, 1], [0.1]),
            ([0, 2], [0.1, 0.9]),
            ([0, 1], [0.1, float("nan")]),
        ):
            with self.assertRaises(ValueError):
                choose_threshold(labels, scores)
        for threshold in (True, "0.5", float("nan"), float("inf"), -1, 2):
            with self.assertRaises(ValueError):
                binary_metrics([0, 1], [0.2, 0.8], threshold)

    def test_fixed_report_does_not_claim_validation_selection(self):
        rows = [dict(source=f"{i}.png", sha256=str(i), group=str(i), label=i) for i in (0, 1)]
        report = evaluation_report(rows, {"a": [0.2, 0.8]}, ["a", "b"], threshold=0.9)
        self.assertEqual("fixed", report["threshold_provenance"]["mode"])
        self.assertFalse(report["threshold_provenance"]["selection_verified"])
        self.assertNotIn("selected", report["ensemble_policy"])
        self.assertEqual("image", report["metric_unit"])
        self.assertEqual(2, report["support"]["declared_group_count"])


class SelectionContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        image_fixture(self.root / "source")
        self.data = self.root / "split"
        self.manifest = create_split(self.root / "source", self.data, independent_images=True)
        self.models = [self.root / "a.keras", self.root / "b.keras"]
        for index, path in enumerate(self.models):
            path.write_bytes(f"synthetic model bytes {index}".encode())
            save_metadata(path, self.manifest, size=(4, 4), architecture="fixture", seed=42)
        self.scores = {digest(self.models[0]): (0.2, 0.4), digest(self.models[1]): (0.6, 0.8)}

    def fake_predictions(self, records, root, manifest, split, *, batch_size):
        rows = [row for row in manifest["rows"] if row["split"] == split]
        predictions, identities = {}, {}
        for index, (_, _, identity) in enumerate(records):
            name = f"model_{index + 1}"
            predictions[name] = [self.scores[identity["sha256"]][row["label"]] for row in rows]
            identities[name] = identity
        return rows, predictions, identities

    def make_selection(self):
        with patch.object(
            evaluate, "_predict_models", side_effect=self.fake_predictions
        ) as predict:
            artifact = select_models(self.models, self.data)
        self.assertEqual("validation", predict.call_args.args[3])
        path = self.root / "selection.json"
        evaluate.write_report(artifact, path)
        return artifact, path

    def test_empty_repeated_path_and_duplicate_bytes_fail_without_tensorflow(self):
        duplicate = self.root / "copy.keras"
        shutil.copyfile(self.models[0], duplicate)
        shutil.copyfile(metadata_path(self.models[0]), metadata_path(duplicate))
        with (
            patch.dict(sys.modules, {"tensorflow": None}),
            patch.object(evaluate, "_predict_models") as predict,
        ):
            for models in ([], [self.models[0], self.models[0]], [self.models[0], duplicate]):
                with self.assertRaises(ValueError):
                    evaluate.evaluate_models(models, self.data)
                with self.assertRaises(ValueError):
                    select_models(models, self.data)
            predict.assert_not_called()

    def test_selection_uses_validation_and_reorders_by_hash(self):
        artifact, path = self.make_selection()
        with patch.object(
            evaluate, "_predict_models", side_effect=self.fake_predictions
        ) as predict:
            reversed_artifact = select_models(self.models[::-1], self.data)
            report = evaluate.evaluate_models(self.models[::-1], self.data, selection=path)
        self.assertEqual(artifact, reversed_artifact)
        self.assertEqual(["validation", "test"], [call.args[3] for call in predict.call_args_list])
        self.assertEqual(0.8, report["metrics"]["model_1"]["threshold"])
        self.assertEqual(0.4, report["metrics"]["model_2"]["threshold"])
        self.assertEqual(0.5, report["metrics"]["ensemble"]["threshold"])
        self.assertEqual("validation_selection", report["threshold_provenance"]["mode"])
        self.assertEqual(digest(path), report["threshold_provenance"]["artifact_sha256"])

    def test_changed_decision_policy_or_evidence_rejected_before_prediction(self):
        artifact, path = self.make_selection()
        first_hash = digest(self.models[0])
        mutations = (
            lambda a: a["decisions"][first_hash].update(threshold=0.9),
            lambda a: a.update(objective="accuracy"),
            lambda a: a.update(tie_break="different"),
            lambda a: a.update(candidate_policy="different"),
            lambda a: a.update(split="test"),
            lambda a: a.update(class_names=a["class_names"][::-1]),
            lambda a: a["validation_probabilities"][first_hash].pop(),
            lambda a: a.update(validation_sample_count=1000),
        )
        for mutate in mutations:
            changed = copy.deepcopy(artifact)
            mutate(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with patch.object(evaluate, "_predict_models") as predict:
                with self.assertRaises(ValueError):
                    evaluate.evaluate_models(self.models, self.data, selection=path)
                predict.assert_not_called()

    def test_model_metadata_manifest_and_member_set_are_bound(self):
        artifact, path = self.make_selection()
        with patch.object(evaluate, "_predict_models") as predict:
            with self.assertRaises(ValueError):
                evaluate.evaluate_models(self.models[:1], self.data, selection=path)
            predict.assert_not_called()
        _, manifest, records = evaluate._prepare_inputs(self.models, self.data, batch_size=32)
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest["seed"] += 1
        with self.assertRaises(ValueError):
            load_selection(path, changed_manifest, records)
        sidecar = metadata_path(self.models[0])
        metadata = json.loads(sidecar.read_text())
        metadata["audit_note"] = "metadata changed without changing model bytes"
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        with patch.object(evaluate, "_predict_models") as predict:
            with self.assertRaises(ValueError):
                evaluate.evaluate_models(self.models, self.data, selection=path)
            predict.assert_not_called()
        # Restoring metadata and then changing model bytes must also fail.
        save_metadata(self.models[0], self.manifest, size=(4, 4), architecture="fixture", seed=42)
        self.models[0].write_bytes(b"new model artifact")
        save_metadata(self.models[0], self.manifest, size=(4, 4), architecture="fixture", seed=42)
        with patch.object(evaluate, "_predict_models") as predict:
            with self.assertRaises(ValueError):
                evaluate.evaluate_models(self.models, self.data, selection=path)
            predict.assert_not_called()

    def test_fixed_threshold_and_mutual_exclusion(self):
        _, path = self.make_selection()
        with patch.object(evaluate, "_predict_models", side_effect=self.fake_predictions):
            default = evaluate.evaluate_models(self.models, self.data)
            fixed = evaluate.evaluate_models(self.models, self.data, threshold=0.9)
        self.assertEqual("default", default["threshold_provenance"]["source"])
        self.assertEqual("caller", fixed["threshold_provenance"]["source"])
        self.assertFalse(fixed["threshold_provenance"]["selection_verified"])
        with patch.object(evaluate, "_predict_models") as predict:
            with self.assertRaises(ValueError):
                evaluate.evaluate_models(self.models, self.data, threshold=0.5, selection=path)
            with self.assertRaises(ValueError):
                evaluate.evaluate_models(self.models, self.data, threshold=float("nan"))
            predict.assert_not_called()

    def test_selection_and_reports_refuse_overwrite(self):
        artifact, path = self.make_selection()
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            evaluate.write_report(artifact, path)
        self.assertEqual(before, path.read_bytes())
        invalid = self.root / "invalid.json"
        with self.assertRaises(ValueError):
            evaluate.write_report({"invalid": float("nan")}, invalid)
        self.assertFalse(invalid.exists())


@unittest.skipUnless(
    os.environ.get("RUN_ML_TESTS") == "1", "Set RUN_ML_TESTS=1 for TensorFlow smoke tests"
)
class SelectionRuntimeTests(unittest.TestCase):
    def test_real_validation_selection_and_test_evaluation(self):
        import numpy as np
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_fixture(root / "source")
            manifest = create_split(root / "source", root / "split", independent_images=True)
            paths = []
            try:
                for index, (negative, positive) in enumerate(((0.2, 0.4), (0.6, 0.8))):
                    model = tf.keras.Sequential(
                        [
                            tf.keras.layers.Input(shape=(4, 4, 3)),
                            tf.keras.layers.Rescaling(1.0 / 255),
                            tf.keras.layers.GlobalAveragePooling2D(),
                            tf.keras.layers.Dense(1, activation="sigmoid"),
                        ]
                    )
                    intercept = math.log(negative / (1 - negative))
                    slope = (math.log(positive / (1 - positive)) - intercept) / (180 / 255)
                    model.layers[-1].set_weights(
                        [
                            np.asarray([[slope], [0], [0]], dtype=np.float32),
                            np.asarray([intercept], dtype=np.float32),
                        ]
                    )
                    path = root / f"{index}.keras"
                    model.save(path)
                    save_metadata(
                        path, manifest, size=(4, 4), architecture="synthetic_fixed_weights", seed=42
                    )
                    paths.append(path)
                with patch.object(evaluate, "dataset", wraps=evaluate.dataset) as datasets:
                    artifact = select_models(paths, root / "split", batch_size=2)
                self.assertEqual(
                    ["validation", "validation"], [call.args[2] for call in datasets.call_args_list]
                )
                selection = root / "selection.json"
                evaluate.write_report(artifact, selection)
                with patch.object(evaluate, "dataset", wraps=evaluate.dataset) as datasets:
                    report = evaluate.evaluate_models(
                        paths[::-1], root / "split", batch_size=2, selection=selection
                    )
                self.assertEqual(
                    ["test", "test"], [call.args[2] for call in datasets.call_args_list]
                )
                self.assertAlmostEqual(0.8, report["metrics"]["model_1"]["threshold"], places=5)
                self.assertAlmostEqual(0.4, report["metrics"]["model_2"]["threshold"], places=5)
                self.assertEqual(0.5, report["metrics"]["ensemble"]["threshold"])
                self.assertTrue(report["threshold_provenance"]["selection_verified"])
                self.assertEqual(
                    [row["source"] for row in manifest["rows"] if row["split"] == "test"],
                    [row["sample_id"] for row in report["samples"]],
                )
            finally:
                tf.keras.backend.clear_session()


if __name__ == "__main__":
    unittest.main()
