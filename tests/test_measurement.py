"""Synthetic arithmetic/oracle checks for frozen-run measurement; no TensorFlow."""

import copy
import hashlib
import json
import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from thyroid_poc import measurement
from thyroid_poc.metrics import evaluation_report


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture(labels=(0, 0, 1, 1), groups=("a", "a", "b", "c")):
    rows = []
    for split, split_labels in (("train", (0, 1, 1, 1)), ("validation", (0, 1)), ("test", labels)):
        for index, label in enumerate(split_labels):
            identity = f"{split}-{index}"
            rows.append(
                dict(
                    source=identity,
                    sha256=sha(identity),
                    label=label,
                    split=split,
                    group=groups[index] if split == "test" else identity,
                )
            )
    return dict(
        schema_version=1,
        class_names=["lower", "higher"],
        source_fingerprint=sha("source"),
        seed=42,
        rows=rows,
    )


def run(manifest, seed, probabilities, threshold):
    rows = [row for row in manifest["rows"] if row["split"] == "test"]
    report = evaluation_report(
        rows,
        {"model_1": probabilities},
        manifest["class_names"],
        threshold=threshold,
        threshold_provenance=dict(
            mode="validation_selection",
            selection_verified=True,
            split="validation",
            objective="balanced_accuracy",
            artifact_sha256=sha(f"selection-{seed}"),
        ),
    )
    report.update(
        split="test",
        source_fingerprint=manifest["source_fingerprint"],
        manifest_sha256=sha(json.dumps(manifest, sort_keys=True)),
        models={
            "model_1": dict(
                sha256=sha(f"model-{seed}"),
                metadata_sha256=sha(f"metadata-{seed}"),
                architecture="fixture",
            )
        },
    )
    return dict(seed=seed, report=report)


def two_runs(manifest):
    return [
        run(manifest, 17, [0.1, 0.3, 0.45, 0.8], 0.4),
        run(manifest, 29, [0.1, 0.6, 0.7, 0.9], 0.8),
    ]


def bootstrap_oracle(manifest, runs, seed, repetitions):
    """Independent NumPy metric/quantile implementation, same specified RNG draws."""
    rows = [row for row in manifest["rows"] if row["split"] == "test"]
    labels = np.asarray([row["label"] for row in rows])
    groups = [
        [index for index, row in enumerate(rows) if row["group"] == group]
        for group in sorted({row["group"] for row in rows})
    ]
    selected, fixed = [], []
    for item in sorted(runs, key=lambda item: item["seed"]):
        report = item["report"]
        scores = np.asarray([sample["probabilities"]["model_1"] for sample in report["samples"]])
        selected.append(scores >= report["metrics"]["model_1"]["threshold"])
        fixed.append(scores >= 0.5)
    values = {
        name: []
        for name in (
            "mean_balanced_accuracy",
            "mean_fixed_threshold_balanced_accuracy",
            "mean_paired_delta_vs_fixed",
            "mean_delta_vs_baseline",
        )
    }
    rng = random.Random(seed)
    invalid, attempted = 0, 0
    while len(values["mean_balanced_accuracy"]) < repetitions:
        attempted += 1
        chosen_groups = [rng.randrange(len(groups)) for _ in range(len(groups))]
        indices = np.concatenate([groups[group] for group in chosen_groups])
        truth = labels[indices]
        if len(np.unique(truth)) < 2:
            invalid += 1
            continue

        def score(prediction):
            actual = prediction[indices]
            return (np.mean(actual[truth == 1]) + np.mean(~actual[truth == 0])) / 2

        left = np.asarray([score(prediction) for prediction in selected])
        right = np.asarray([score(prediction) for prediction in fixed])
        values["mean_balanced_accuracy"].append(float(left.mean()))
        values["mean_fixed_threshold_balanced_accuracy"].append(float(right.mean()))
        values["mean_paired_delta_vs_fixed"].append(float((left - right).mean()))
        values["mean_delta_vs_baseline"].append(float((left - 0.5).mean()))
    intervals = {
        name: np.quantile(scores, [0.025, 0.975], method="linear")
        for name, scores in values.items()
    }
    return intervals, invalid, attempted


class MeasurementArithmeticTests(unittest.TestCase):
    def test_recomputed_scores_sample_variance_and_fixed_comparison(self):
        manifest = fixture()
        runs = two_runs(manifest)
        # Cached report summaries are not trusted as measurements.
        runs[0]["report"]["metrics"]["model_1"]["balanced_accuracy"] = -100
        summary = measurement.summarize_measurement(manifest, runs, bootstrap_repetitions=40)
        self.assertEqual([1.0, 0.75], [item["balanced_accuracy"] for item in summary["per_seed"]])
        self.assertEqual(
            [0.75, 0.75],
            [item["fixed_threshold_balanced_accuracy"] for item in summary["per_seed"]],
        )
        self.assertEqual(
            [0.25, 0.0], [item["paired_delta_vs_fixed"] for item in summary["per_seed"]]
        )
        statistics = summary["across_seeds"]["balanced_accuracy"]
        self.assertEqual(0.875, statistics["mean"])
        self.assertEqual(0.03125, statistics["sample_variance"])
        self.assertAlmostEqual(math.sqrt(0.03125), statistics["sample_sd"])
        self.assertEqual(1, statistics["ddof"])
        self.assertEqual(0.75, statistics["min"])
        self.assertEqual(1.0, statistics["max"])
        self.assertEqual(0.125, summary["across_seeds"]["paired_delta_vs_fixed"]["mean"])
        self.assertEqual(0.75, summary["baseline"]["positive_prevalence"])
        self.assertEqual(0.5, summary["baseline"]["balanced_accuracy"])
        self.assertEqual(
            [0.25, 0.25], summary["test_resolution"]["one_error_balanced_accuracy_change_by_class"]
        )
        json.dumps(summary, allow_nan=False)

    def test_group_bootstrap_matches_independent_oracle_with_mixed_unequal_groups(self):
        manifest = fixture((0, 0, 1, 0, 1, 1), ("a", "a", "b", "c", "c", "c"))
        runs = [
            run(manifest, 17, [0.2, 0.6, 0.7, 0.3, 0.45, 0.9], 0.4),
            run(manifest, 29, [0.1, 0.4, 0.55, 0.7, 0.8, 0.6], 0.65),
        ]
        seed, repetitions = 19, 150
        expected, invalid, attempted = bootstrap_oracle(manifest, runs, seed, repetitions)
        result = measurement.summarize_measurement(
            manifest, runs, bootstrap_seed=seed, bootstrap_repetitions=repetitions
        )["group_bootstrap"]
        self.assertEqual("complete", result["status"])
        self.assertEqual(invalid, result["invalid_single_class_draws"])
        self.assertEqual(attempted, result["attempted_draws"])
        for name, interval in expected.items():
            actual = result["intervals"][name]
            np.testing.assert_allclose(
                [actual["lower"], actual["upper"]], interval, atol=1e-14, rtol=0
            )

    def test_zero_paired_effect_when_thresholds_are_equal(self):
        manifest = fixture()
        runs = [
            run(manifest, 17, [0.1, 0.8, 0.3, 0.9], 0.5),
            run(manifest, 29, [0.7, 0.2, 0.8, 0.4], 0.5),
        ]
        result = measurement.summarize_measurement(manifest, runs, bootstrap_repetitions=80)
        self.assertEqual([0.0, 0.0], [row["paired_delta_vs_fixed"] for row in result["per_seed"]])
        self.assertEqual(
            dict(lower=0.0, upper=0.0),
            result["group_bootstrap"]["intervals"]["mean_paired_delta_vs_fixed"],
        )

    def test_seed_count_is_not_extra_test_support_or_narrower_group_interval(self):
        manifest = fixture((0, 0, 1, 0, 1, 1), ("a", "a", "b", "c", "c", "c"))
        scores = [0.2, 0.8, 0.7, 0.6, 0.4, 0.9]
        small = [run(manifest, seed, scores, 0.5) for seed in (17, 29)]
        larger = [run(manifest, seed, scores, 0.5) for seed in (17, 29, 43, 71, 101)]
        a = measurement.summarize_measurement(manifest, small, bootstrap_repetitions=100)
        b = measurement.summarize_measurement(manifest, larger, bootstrap_repetitions=100)
        self.assertEqual(a["test_support"], b["test_support"])
        self.assertEqual(
            a["group_bootstrap"]["attempted_draws"], b["group_bootstrap"]["attempted_draws"]
        )
        for name, interval in a["group_bootstrap"]["intervals"].items():
            other = b["group_bootstrap"]["intervals"][name]
            self.assertAlmostEqual(interval["lower"], other["lower"], places=14)
            self.assertAlmostEqual(interval["upper"], other["upper"], places=14)

    def test_training_prior_does_not_depend_on_validation_or_test_prevalence(self):
        first = fixture()
        second = fixture((0, 0, 0, 1), ("a", "a", "b", "c"))
        for row in second["rows"]:
            if row["split"] == "validation":
                row["label"] = 0
        a = measurement.summarize_measurement(first, two_runs(first), bootstrap_repetitions=20)
        b = measurement.summarize_measurement(second, two_runs(second), bootstrap_repetitions=20)
        self.assertEqual(a["baseline"], b["baseline"])

    def test_run_order_replay_and_input_preservation(self):
        manifest = fixture()
        runs = two_runs(manifest)
        before = copy.deepcopy((manifest, runs))
        a = measurement.summarize_measurement(manifest, runs, bootstrap_repetitions=40)
        b = measurement.summarize_measurement(manifest, runs[::-1], bootstrap_repetitions=40)
        self.assertEqual(a, b)
        self.assertEqual(before, (manifest, runs))

    def test_exhausted_missing_class_draws_are_bounded_without_partial_intervals(self):
        manifest = fixture()
        with patch.object(
            measurement.random, "Random", return_value=SimpleNamespace(randrange=lambda n: 0)
        ):
            result = measurement.summarize_measurement(
                manifest, two_runs(manifest), bootstrap_repetitions=2
            )
        bootstrap = result["group_bootstrap"]
        self.assertEqual("insufficient_valid_draws", bootstrap["status"])
        self.assertEqual(40, bootstrap["attempted_draws"])
        self.assertEqual(40, bootstrap["invalid_single_class_draws"])
        self.assertEqual(0, bootstrap["valid_draws"])
        self.assertIsNone(bootstrap["intervals"])


class MeasurementInputTests(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture()
        self.runs = two_runs(self.manifest)

    def rejects(self, manifest=None, runs=None, **kwargs):
        with self.assertRaises(ValueError):
            measurement.summarize_measurement(
                self.manifest if manifest is None else manifest,
                self.runs if runs is None else runs,
                bootstrap_repetitions=5,
                **kwargs,
            )

    def test_seed_and_bootstrap_arguments(self):
        self.rejects(runs=[])
        self.rejects(runs=self.runs[:1])
        repeated = copy.deepcopy(self.runs)
        repeated[1]["seed"] = repeated[0]["seed"]
        self.rejects(runs=repeated)
        repeated[1]["seed"] = True
        self.rejects(runs=repeated)
        self.rejects(bootstrap_seed=True)
        for repetitions in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                measurement.summarize_measurement(
                    self.manifest, self.runs, bootstrap_repetitions=repetitions
                )

    def test_each_sample_identity_and_probability_contract(self):
        mutations = (
            lambda s: s.update(sample_id="different"),
            lambda s: s.update(sha256=sha("different")),
            lambda s: s.update(group="different"),
            lambda s: s.update(true_label=1),
            lambda s: s.update(true_label=False),
            lambda s: s.update(true_class="higher"),
            lambda s: s.update(probabilities={"model_2": 0.2}),
            lambda s: s["probabilities"].update(ensemble=0.2),
        )
        for mutate in mutations:
            changed = copy.deepcopy(self.runs)
            mutate(changed[0]["report"]["samples"][0])
            self.rejects(runs=changed)
        for probability in (True, ".2", -0.1, 1.1, float("nan"), float("inf")):
            changed = copy.deepcopy(self.runs)
            changed[0]["report"]["samples"][0]["probabilities"]["model_1"] = probability
            self.rejects(runs=changed)

    def test_sample_order_count_and_single_model_identity(self):
        for transform in (
            lambda samples: samples[::-1],
            lambda samples: samples[:-1],
            lambda samples: [samples[0], *samples[1:-1], samples[0]],
        ):
            changed = copy.deepcopy(self.runs)
            changed[0]["report"]["samples"] = transform(changed[0]["report"]["samples"])
            self.rejects(runs=changed)
        changed = copy.deepcopy(self.runs)
        changed[0]["report"]["models"]["model_2"] = changed[0]["report"]["models"]["model_1"]
        self.rejects(runs=changed)

    def test_manifest_and_selection_report_bindings(self):
        mutations = (
            lambda r: r.update(split="validation"),
            lambda r: r.update(class_names=["higher", "lower"]),
            lambda r: r.update(positive_class="lower"),
            lambda r: r.update(source_fingerprint=sha("different")),
            lambda r: r.update(manifest_sha256=sha("different")),
            lambda r: r.update(metric_unit="patient"),
            lambda r: r.update(ensemble_policy="unweighted arithmetic mean"),
            lambda r: r["threshold_provenance"].update(mode="fixed"),
            lambda r: r["threshold_provenance"].update(selection_verified=False),
            lambda r: r["threshold_provenance"].update(artifact_sha256="invalid"),
            lambda r: r["metrics"]["model_1"].update(threshold=float("nan")),
        )
        for mutate in mutations:
            changed = copy.deepcopy(self.runs)
            mutate(changed[0]["report"])
            self.rejects(runs=changed)

    def test_unresolved_groups_class_support_and_partition_leakage(self):
        changed = copy.deepcopy(self.manifest)
        test = [row for row in changed["rows"] if row["split"] == "test"]
        test[1]["sha256"] = test[0]["sha256"]
        test[1]["group"] = "linked-other-group"
        self.rejects(manifest=changed)
        changed = copy.deepcopy(self.manifest)
        for row in changed["rows"]:
            if row["split"] == "test":
                row["group"] = "one-group"
        self.rejects(manifest=changed)
        changed = copy.deepcopy(self.manifest)
        changed["rows"][0]["group"] = "a"
        self.rejects(manifest=changed)
        changed = copy.deepcopy(self.manifest)
        for row in changed["rows"]:
            if row["split"] == "test":
                row["label"] = 1
        self.rejects(manifest=changed)


if __name__ == "__main__":
    unittest.main()
