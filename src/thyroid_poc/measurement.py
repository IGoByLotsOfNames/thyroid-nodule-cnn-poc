"""Summarize frozen single-model runs without training or importing TensorFlow.

Image-level balanced accuracy is recomputed from aligned predictions. The group
bootstrap is paired across fixed fitted models and comparators; training seeds
are never added to the number of independent test observations.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from numbers import Real

from .data import group_support, manifest_fingerprint
from .metrics import validate_threshold


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a canonical SHA-256 identity")
    return value


def _manifest_rows(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Expected a dataset manifest object")
    classes = manifest.get("class_names")
    if (
        not isinstance(classes, list)
        or len(classes) != 2
        or any(not isinstance(name, str) or not name for name in classes)
        or len(set(classes)) != 2
    ):
        raise ValueError("Manifest needs two ordered class names")
    _hash(manifest.get("source_fingerprint"))
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest needs sample rows")
    sources, group_splits, hash_splits, hash_labels = set(), {}, {}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid manifest row")
        source, group, split, label = (
            row.get(key) for key in ("source", "group", "split", "label")
        )
        if not isinstance(source, str) or not source or source in sources:
            raise ValueError("Manifest sample identities must be nonempty and unique")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Manifest groups must be nonempty strings")
        if (
            split not in ("train", "validation", "test")
            or type(label) is not int
            or label not in (0, 1)
        ):
            raise ValueError("Invalid manifest split or label")
        image_hash = _hash(row.get("sha256"))
        sources.add(source)
        for key, registry in ((group, group_splits), (image_hash, hash_splits)):
            if key in registry and registry[key] != split:
                raise ValueError("Manifest group or exact duplicate crosses partitions")
            registry[key] = split
        if image_hash in hash_labels and hash_labels[image_hash] != label:
            raise ValueError("Duplicate bytes have conflicting labels")
        hash_labels[image_hash] = label
    training = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if any({row["label"] for row in part} != {0, 1} for part in (training, test)):
        raise ValueError("Training and test partitions must each contain both classes")
    hash_groups = {}
    for row in test:
        if row["sha256"] in hash_groups and hash_groups[row["sha256"]] != row["group"]:
            raise ValueError(
                "Resolve exact-duplicate links between test groups before bootstrapping"
            )
        hash_groups[row["sha256"]] = row["group"]
    if len({row["group"] for row in test}) < 2:
        raise ValueError("Group uncertainty requires at least two declared test groups")
    return classes, training, test


def _validated_runs(manifest, classes, test, runs):
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError("Supply at least two distinct training-seed runs")
    seeds, result, report_manifest_hash = set(), [], None
    for run in runs:
        if not isinstance(run, dict) or type(run.get("seed")) is not int or run["seed"] in seeds:
            raise ValueError("Run seeds must be distinct integers")
        seeds.add(run["seed"])
        report = run.get("report")
        if not isinstance(report, dict):
            raise ValueError("Each run needs an evaluation report")
        if (
            report.get("split") != "test"
            or report.get("class_names") != classes
            or report.get("positive_class") != classes[1]
            or report.get("source_fingerprint") != manifest["source_fingerprint"]
            or report.get("metric_unit") != "image"
        ):
            raise ValueError("Evaluation report does not match the test/class/source contract")
        current_manifest_hash = _hash(report.get("manifest_sha256"))
        if report_manifest_hash is not None and current_manifest_hash != report_manifest_hash:
            raise ValueError("Runs refer to different serialized split manifests")
        report_manifest_hash = current_manifest_hash
        models, metrics = report.get("models"), report.get("metrics")
        if (
            not isinstance(models, dict)
            or set(models) != {"model_1"}
            or not isinstance(metrics, dict)
            or set(metrics) != {"model_1"}
            or report.get("ensemble_policy") != "single model"
        ):
            raise ValueError("Measurement requires exactly one unambiguous model_1")
        identity = models["model_1"]
        model_hash = _hash(identity.get("sha256")) if isinstance(identity, dict) else _hash(None)
        _hash(identity.get("metadata_sha256"))
        provenance = report.get("threshold_provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("mode") != "validation_selection"
            or provenance.get("selection_verified") is not True
            or provenance.get("split") != "validation"
            or provenance.get("objective") != "balanced_accuracy"
        ):
            raise ValueError("Measurement requires validation-selected threshold provenance")
        selection_hash = _hash(provenance.get("artifact_sha256"))
        if not isinstance(metrics["model_1"], dict):
            raise ValueError("Missing single-model threshold")
        threshold = validate_threshold(metrics["model_1"].get("threshold"))
        samples = report.get("samples")
        if not isinstance(samples, list) or len(samples) != len(test):
            raise ValueError("Evaluation samples do not cover the frozen test manifest")
        probabilities = []
        for row, sample in zip(test, samples):
            if not isinstance(sample, dict) or (
                sample.get("sample_id") != row["source"]
                or sample.get("sha256") != row["sha256"]
                or sample.get("group") != row["group"]
                or type(sample.get("true_label")) is not int
                or sample["true_label"] != row["label"]
                or sample.get("true_class") != classes[row["label"]]
            ):
                raise ValueError(
                    "Evaluation sample order or identity disagrees with the test manifest"
                )
            values = sample.get("probabilities")
            if not isinstance(values, dict) or set(values) != {"model_1"}:
                raise ValueError("Each sample needs exactly one model_1 probability")
            probability = values["model_1"]
            if (
                isinstance(probability, bool)
                or not isinstance(probability, Real)
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Expected finite numeric probabilities in [0,1]")
            probabilities.append(float(probability))
        result.append(
            dict(
                seed=run["seed"],
                model_sha256=model_hash,
                selection_artifact_sha256=selection_hash,
                selected_threshold=threshold,
                selected=[int(p >= threshold) for p in probabilities],
                fixed=[int(p >= 0.5) for p in probabilities],
            )
        )
    return sorted(result, key=lambda item: item["seed"]), report_manifest_hash


def _balanced_accuracy(labels, predictions, indices):
    positives = sum(labels[index] for index in indices)
    negatives = len(indices) - positives
    if not positives or not negatives:
        return None
    true_positive = sum(labels[index] == 1 and predictions[index] == 1 for index in indices)
    true_negative = sum(labels[index] == 0 and predictions[index] == 0 for index in indices)
    return (true_positive / positives + true_negative / negatives) / 2


def _statistics(values):
    variance = statistics.variance(values)
    return dict(
        count=len(values),
        mean=statistics.fmean(values),
        sample_sd=math.sqrt(variance),
        sample_variance=variance,
        min=min(values),
        max=max(values),
        ddof=1,
    )


def _percentile_interval(values):
    ordered = sorted(values)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return dict(lower=percentile(0.025), upper=percentile(0.975))


def _group_bootstrap(test, labels, runs, baseline_predictions, seed, repetitions):
    groups = {}
    for index, row in enumerate(test):
        groups.setdefault(row["group"], []).append(index)
    ordered_groups = [groups[key] for key in sorted(groups)]
    rng = random.Random(seed)
    values = {
        key: []
        for key in (
            "mean_balanced_accuracy",
            "mean_fixed_threshold_balanced_accuracy",
            "mean_paired_delta_vs_fixed",
            "mean_delta_vs_baseline",
        )
    }
    valid, invalid, attempts = 0, 0, 0
    maximum = 20 * repetitions
    while valid < repetitions and attempts < maximum:
        attempts += 1
        indices = [
            index
            for _ in ordered_groups
            for index in ordered_groups[rng.randrange(len(ordered_groups))]
        ]
        baseline = _balanced_accuracy(labels, baseline_predictions, indices)
        if baseline is None:
            invalid += 1
            continue
        selected = [_balanced_accuracy(labels, run["selected"], indices) for run in runs]
        fixed = [_balanced_accuracy(labels, run["fixed"], indices) for run in runs]
        values["mean_balanced_accuracy"].append(statistics.fmean(selected))
        values["mean_fixed_threshold_balanced_accuracy"].append(statistics.fmean(fixed))
        values["mean_paired_delta_vs_fixed"].append(
            statistics.fmean(left - right for left, right in zip(selected, fixed))
        )
        values["mean_delta_vs_baseline"].append(
            statistics.fmean(value - baseline for value in selected)
        )
        valid += 1
    return dict(
        status="complete" if valid == repetitions else "insufficient_valid_draws",
        seed=seed,
        unit="declared_test_group",
        groups_per_draw=len(ordered_groups),
        repetitions_requested=repetitions,
        valid_draws=valid,
        invalid_single_class_draws=invalid,
        attempted_draws=attempts,
        maximum_attempts=maximum,
        confidence_level=0.95,
        method="percentile with linear interpolation at 2.5% and 97.5%",
        intervals={key: _percentile_interval(items) for key, items in values.items()}
        if valid == repetitions
        else None,
        conditioning="Fixed fitted models and thresholds; resamples retaining both true classes.",
        pairing="The same whole-group draws, with replacement and full image multiplicity, are used for every seed and comparator.",
    )


def summarize_measurement(manifest, runs, *, bootstrap_seed=1729, bootstrap_repetitions=2000):
    """Recompute paired image-level measurements for at least two frozen seed runs.

    This checks report alignment and declared provenance, not model or selection
    files. The experiment runner must verify those artifact bindings separately.
    """
    if (
        type(bootstrap_seed) is not int
        or type(bootstrap_repetitions) is not int
        or bootstrap_repetitions <= 0
    ):
        raise ValueError("Bootstrap seed must be integer and repetitions a positive integer")
    classes, training, test = _manifest_rows(manifest)
    validated, report_manifest_hash = _validated_runs(manifest, classes, test, runs)
    labels = [row["label"] for row in test]
    indices = list(range(len(test)))
    prior = sum(row["label"] for row in training) / len(training)
    baseline_predictions = [int(prior >= 0.5)] * len(test)
    baseline_score = _balanced_accuracy(labels, baseline_predictions, indices)
    per_seed = []
    for run in validated:
        selected = _balanced_accuracy(labels, run["selected"], indices)
        fixed = _balanced_accuracy(labels, run["fixed"], indices)
        per_seed.append(
            dict(
                seed=run["seed"],
                model_sha256=run["model_sha256"],
                selection_artifact_sha256=run["selection_artifact_sha256"],
                selected_threshold=run["selected_threshold"],
                balanced_accuracy=selected,
                fixed_threshold_balanced_accuracy=fixed,
                paired_delta_vs_fixed=selected - fixed,
                delta_vs_baseline=selected - baseline_score,
            )
        )
    metric_names = (
        "balanced_accuracy",
        "fixed_threshold_balanced_accuracy",
        "paired_delta_vs_fixed",
        "delta_vs_baseline",
    )
    counts = [labels.count(0), labels.count(1)]
    return dict(
        schema_version=1,
        metric="image_level_balanced_accuracy",
        primary_comparison="validation_selected_threshold_minus_fixed_0.5_on_identical_model_probabilities",
        class_names=classes,
        split_manifest_fingerprint=manifest_fingerprint(manifest),
        report_manifest_sha256=report_manifest_hash,
        seed_count=len(validated),
        test_support=group_support(test),
        baseline=dict(
            name="constant_training_positive_prevalence",
            training_sample_count=len(training),
            training_positive_count=sum(row["label"] for row in training),
            positive_prevalence=prior,
            threshold=0.5,
            predicted_class=int(prior >= 0.5),
            balanced_accuracy=baseline_score,
        ),
        per_seed=per_seed,
        across_seeds={name: _statistics([row[name] for row in per_seed]) for name in metric_names},
        group_bootstrap=_group_bootstrap(
            test, labels, validated, baseline_predictions, bootstrap_seed, bootstrap_repetitions
        ),
        test_resolution=dict(
            images_by_class=counts,
            one_error_balanced_accuracy_change_by_class=[0.5 / n for n in counts],
        ),
        limitations=[
            "Group IDs are not verified independent patients; group resampling assumes approximately independent declared groups.",
            f"The smaller test class has {min(counts)} images; percentile intervals are approximate and may be unstable.",
            "Group-bootstrap intervals condition on these frozen models and do not include retraining, dataset selection, near-duplicate or external-site uncertainty.",
            "Training-seed sample SD and variance (ddof=1) describe observed run variation separately; seeds are not additional patients or test observations.",
            "Selected-versus-fixed comparison must be frozen before inspecting test results; it is not a comparison with historical study figures or earlier label bugs.",
        ],
    )
