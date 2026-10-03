"""Freeze, run and summarize a repeated-seed CPU experiment; no downloads."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import digest, manifest_fingerprint, safe_path, validate_manifest
from .evaluate import write_report

DEFAULT_SEEDS = (17, 29, 43, 71, 101)
FIT_FILES = ("model.keras", "model.metadata.json", "model.history.json", "selection.json")


def _versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "Pillow", "tensorflow", "keras"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _source_hashes():
    return {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def prepare(
    data, output, *, seeds=DEFAULT_SEEDS, epochs=30, batch_size=16, bootstrap_repetitions=2000
):
    """Record choices before any model fitting; does not import TensorFlow."""
    data, output = Path(data).resolve(), Path(output).resolve()
    seeds = list(seeds)
    if (
        len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
    ):
        raise ValueError("Use at least two distinct unsigned 32-bit integer seeds")
    if any(type(value) is not int or value <= 0 for value in (epochs, batch_size)):
        raise ValueError("Epochs and batch size must be positive integers")
    if type(bootstrap_repetitions) is not int or bootstrap_repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required")
    if output.exists():
        raise FileExistsError("Choose a fresh experiment directory")
    if output.is_relative_to(data) or data.is_relative_to(output):
        raise ValueError("Dataset and experiment must not contain each other")
    manifest = validate_manifest(data)
    plan = dict(
        schema_version=1,
        kind="fixed_split_repeated_seed_measurement",
        data_path=Path(os.path.relpath(data, output)).as_posix(),
        data_manifest_sha256=digest(data / "manifest.json"),
        manifest_fingerprint=manifest_fingerprint(manifest),
        source_hashes=_source_hashes(),
        versions=_versions(),
        recipe=dict(
            seeds=seeds,
            architecture="alexnet",
            image_size=[224, 224],
            pretrained=False,
            max_epochs=epochs,
            batch_size=batch_size,
            optimizer="Adam",
            learning_rate=1e-4,
            loss="binary_crossentropy",
            class_weight=None,
            augmentation=None,
            early_stopping=dict(monitor="val_loss", patience=5, restore_best_weights=True),
            cpu_intra_threads=2,
            cpu_inter_threads=1,
        ),
        measurement=dict(
            primary="image_level_balanced_accuracy",
            paired_effect="validation_selected_minus_fixed_0.5",
            baseline="training_positive_prevalence_at_fixed_0.5",
            bootstrap_seed=1729,
            bootstrap_repetitions=bootstrap_repetitions,
            all_seeds_reported=True,
        ),
        machine=dict(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            processor=platform.processor(),
        ),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    output.mkdir(parents=True)
    write_report(plan, output / "plan.json")
    (output / "plan.sha256").write_text(digest(output / "plan.json"), encoding="ascii")
    return plan


def load_plan(output):
    output = Path(output).resolve()
    plan_path = output / "plan.json"
    if digest(plan_path) != (output / "plan.sha256").read_text(encoding="ascii"):
        raise ValueError("Plan changed after preparation; create a new experiment")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["source_hashes"] != _source_hashes() or plan["versions"] != _versions():
        raise ValueError("Code or runtime versions changed after preparation")
    data = (output / plan["data_path"]).resolve()
    manifest = validate_manifest(data)
    if (
        digest(data / "manifest.json") != plan["data_manifest_sha256"]
        or manifest_fingerprint(manifest) != plan["manifest_fingerprint"]
    ):
        raise ValueError("Frozen dataset manifest changed")
    return output, plan, data, manifest


def _checkpoint(output, phase, seed):
    path = output / f"{phase}-{seed}.json"
    entry = json.loads(path.read_text(encoding="utf-8"))
    return _verify_entry(output, entry, phase, seed)


def _verify_entry(output, entry, phase, seed):
    if (
        entry["seed"] != seed
        or entry["phase"] != phase
        or entry["plan_sha256"] != digest(output / "plan.json")
    ):
        raise ValueError("Checkpoint belongs to another experiment")
    folder = safe_path(output, entry["directory"])
    expected = set(FIT_FILES) if phase == "fit" else {"test-report.json"}
    if set(entry["files"]) != expected:
        raise ValueError("Checkpoint artifact inventory is incomplete")
    for name, checksum in entry["files"].items():
        if digest(safe_path(folder, name)) != checksum:
            raise ValueError("Completed experiment artifact changed")
    if phase == "fit":
        metadata = json.loads((folder / "model.metadata.json").read_text(encoding="utf-8"))
        if type(metadata.get("seed")) is not int or metadata["seed"] != seed:
            raise ValueError("Fitted model metadata does not match the planned seed")
    return entry, folder


def _lock(output, seeds):
    # All fitting and validation decisions are sealed before any test prediction.
    refs = {}
    for seed in seeds:
        _checkpoint(output, "fit", seed)
        refs[str(seed)] = digest(output / f"fit-{seed}.json")
    expected = dict(plan_sha256=digest(output / "plan.json"), fit_checkpoints=refs)
    path = output / "evaluation-lock.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Fitted artifacts differ from the pre-test lock")
    else:
        write_report(expected, path)
    return expected


def _worker_environment(seed):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONHASHSEED=str(seed),
        TF_ENABLE_ONEDNN_OPTS="0",
        TF_DETERMINISTIC_OPS="1",
        TF_CPP_MIN_LOG_LEVEL="2",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def _launch(output, phase, seed):
    """Keep failed attempts intact; a retry uses a new directory for the same seed."""
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f"{phase}-{seed}-", dir=attempts))
    command = [
        sys.executable,
        "-B",
        "-m",
        "thyroid_poc.experiment",
        "worker",
        str(output),
        phase,
        str(seed),
        folder.relative_to(output).as_posix(),
    ]
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    with (folder / "process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=_worker_environment(seed),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
        process.stdout.close()
    write_report(
        dict(
            command=command,
            seed=seed,
            phase=phase,
            started_at_utc=started,
            wall_seconds=time.perf_counter() - clock,
            returncode=code,
            timing_scope="worker startup, integrity checks, fitting/selection or test prediction, artifact writes",
        ),
        folder / "process.json",
    )
    if code:
        raise RuntimeError(
            f"Seed {seed} {phase} failed; see {folder / 'process.log'}. Retry the same run command."
        )
    entry = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    if entry["directory"] != folder.relative_to(output).as_posix():
        raise ValueError("Worker bundle does not belong to its allocated attempt")
    _verify_entry(output, entry, phase, seed)
    write_report(entry, output / f"{phase}-{seed}.json")


def worker(output, phase, seed, relative):
    output, plan, data, manifest = load_plan(output)
    seeds = plan["recipe"]["seeds"]
    if seed not in seeds or phase not in ("fit", "test"):
        raise ValueError("Worker must use a planned seed and phase")
    folder = safe_path(output, relative)
    if folder.parent != output / "attempts" or not folder.is_dir():
        raise ValueError("Worker output must be an allocated attempt directory")
    if phase == "fit" and (output / "evaluation-lock.json").exists():
        raise ValueError("Fitting is forbidden after the pre-test lock")
    if phase == "test":
        if not (output / "evaluation-lock.json").exists():
            raise ValueError("All fits must be locked before test prediction")
        _lock(output, seeds)
    expected_env = _worker_environment(seed)
    fixed_keys = (
        "CUDA_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
        "TF_ENABLE_ONEDNN_OPTS",
        "TF_DETERMINISTIC_OPS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    if any(os.environ.get(key) != expected_env[key] for key in fixed_keys):
        raise ValueError(
            "Worker environment does not match the frozen CPU recipe; use the run command"
        )
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.get_visible_devices("GPU"):
        raise ValueError("This frozen experiment requires CPU-only workers")
    if phase == "fit":
        from .selection import select_models
        from .train import train_model

        recipe = plan["recipe"]
        train_model(
            data,
            folder / "model.keras",
            architecture=recipe["architecture"],
            epochs=recipe["max_epochs"],
            batch_size=recipe["batch_size"],
            seed=seed,
            pretrained=recipe["pretrained"],
            size=tuple(recipe["image_size"]),
        )
        tf.keras.backend.clear_session()
        history = json.loads((folder / "model.history.json").read_text(encoding="utf-8"))
        if (
            not history.get("loss")
            or not history.get("val_loss")
            or len(history["loss"]) != len(history["val_loss"])
            or any(not math.isfinite(value) for values in history.values() for value in values)
        ):
            raise ValueError("Training history must contain finite, aligned epoch values")
        selected = select_models([folder / "model.keras"], data, batch_size=recipe["batch_size"])
        write_report(selected, folder / "selection.json")
        details = dict(
            epochs_completed=len(history["loss"]),
            best_validation_loss_epoch=1
            + min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__),
        )
        files = FIT_FILES
    else:
        from .evaluate import evaluate_models

        _, fit_folder = _checkpoint(output, "fit", seed)
        report = evaluate_models(
            [fit_folder / "model.keras"],
            data,
            batch_size=plan["recipe"]["batch_size"],
            selection=fit_folder / "selection.json",
        )
        write_report(report, folder / "test-report.json")
        details, files = {}, ("test-report.json",)
    details.update(
        worker_environment={key: os.environ[key] for key in fixed_keys},
        cpu_intra_threads=tf.config.threading.get_intra_op_parallelism_threads(),
        cpu_inter_threads=tf.config.threading.get_inter_op_parallelism_threads(),
        visible_devices=[device.device_type for device in tf.config.get_visible_devices()],
    )
    entry = dict(
        phase=phase,
        seed=seed,
        plan_sha256=digest(output / "plan.json"),
        directory=folder.relative_to(output).as_posix(),
        files={name: digest(folder / name) for name in files},
        details=details,
    )
    write_report(entry, folder / "bundle.json")


def summarize(output):
    from .measurement import summarize_measurement

    output, plan, _, manifest = load_plan(output)
    seeds = plan["recipe"]["seeds"]
    if not (output / "evaluation-lock.json").exists():
        raise ValueError("Experiment is incomplete; no evaluation lock")
    _lock(output, seeds)
    runs = []
    for seed in seeds:
        _, folder = _checkpoint(output, "test", seed)
        report = json.loads((folder / "test-report.json").read_text(encoding="utf-8"))
        _, fit_folder = _checkpoint(output, "fit", seed)
        if report["models"]["model_1"]["sha256"] != digest(fit_folder / "model.keras"):
            raise ValueError("Test report does not match the planned fitted model")
        if report["threshold_provenance"]["artifact_sha256"] != digest(
            fit_folder / "selection.json"
        ):
            raise ValueError("Test report does not match the frozen threshold selection")
        selected = json.loads((fit_folder / "selection.json").read_text(encoding="utf-8"))
        model_hash = digest(fit_folder / "model.keras")
        if (
            report["metrics"]["model_1"]["threshold"]
            != selected["decisions"][model_hash]["threshold"]
            or report["manifest_sha256"] != plan["data_manifest_sha256"]
            or report["models"]["model_1"]["metadata_sha256"]
            != digest(fit_folder / "model.metadata.json")
        ):
            raise ValueError(
                "Test report threshold, metadata or manifest differs from frozen artifacts"
            )
        runs.append(dict(seed=seed, report=report))
    measurement = plan["measurement"]
    result = summarize_measurement(
        manifest,
        runs,
        bootstrap_seed=measurement["bootstrap_seed"],
        bootstrap_repetitions=measurement["bootstrap_repetitions"],
    )
    result["experiment"] = dict(
        plan_sha256=digest(output / "plan.json"),
        seeds=seeds,
        evaluation_lock_sha256=digest(output / "evaluation-lock.json"),
        test_checkpoint_sha256={str(seed): digest(output / f"test-{seed}.json") for seed in seeds},
    )
    result["experiment"]["attempts"] = [
        dict(
            directory=folder.relative_to(output).as_posix(),
            process=json.loads((folder / "process.json").read_text(encoding="utf-8"))
            if (folder / "process.json").exists()
            else {"status": "interrupted_or_unrecorded"},
        )
        for folder in sorted((output / "attempts").iterdir())
        if folder.is_dir()
    ]
    path = output / "results.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing results disagree with verified experiment artifacts")
    else:
        write_report(result, path)
    return result


def run(output):
    output, plan, _, _ = load_plan(output)
    seeds = plan["recipe"]["seeds"]
    for phase in ("fit", "test"):
        if phase == "test":
            _lock(output, seeds)
        for index, seed in enumerate(seeds, 1):
            print(f"{phase}: seed {seed} ({index}/{len(seeds)})", flush=True)
            if (output / f"{phase}-{seed}.json").exists():
                _checkpoint(output, phase, seed)
            else:
                if phase == "fit" and (output / "evaluation-lock.json").exists():
                    raise ValueError(
                        "Cannot refit a missing seed after test evaluation was unlocked"
                    )
                _launch(output, phase, seed)
    return summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser(
        "prepare", help="Freeze a fresh experiment without training"
    )
    prepare_command.add_argument("data", type=Path)
    prepare_command.add_argument("output", type=Path)
    prepare_command.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    prepare_command.add_argument("--epochs", type=int, default=30)
    prepare_command.add_argument("--batch-size", type=int, default=16)
    prepare_command.add_argument("--bootstrap-repetitions", type=int, default=2000)
    for name in ("run", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("output", type=Path)
    child = commands.add_parser("worker", help=argparse.SUPPRESS)
    child.add_argument("output", type=Path)
    child.add_argument("phase", choices=("fit", "test"))
    child.add_argument("seed", type=int)
    child.add_argument("relative")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(
            args.data,
            args.output,
            seeds=args.seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
        print(f"Prepared only; no model fitted. Plan: {args.output / 'plan.json'}")
    elif args.command == "worker":
        worker(args.output, args.phase, args.seed, args.relative)
    else:
        result = run(args.output) if args.command == "run" else summarize(args.output)
        print(
            json.dumps(
                dict(
                    across_seeds=result["across_seeds"], group_bootstrap=result["group_bootstrap"]
                ),
                indent=2,
            )
        )
        print(f"Full measurement and provenance: {args.output / 'results.json'}")
        if result["group_bootstrap"]["status"] != "complete":
            raise SystemExit(
                "Uncertainty estimation incomplete; retain the recorded failed-draw counts."
            )


if __name__ == "__main__":
    main()
