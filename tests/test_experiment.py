"""Experiment orchestration with synthetic data and fake workers; never trains."""

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_selection import image_fixture

from thyroid_poc import evaluate, experiment
from thyroid_poc.artifacts import save_metadata
from thyroid_poc.data import create_split, digest
from thyroid_poc.evaluate import write_report
from thyroid_poc.selection import select_models


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="experiment fixtures ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        image_fixture(self.root / "source")
        self.data = self.root / "split data"
        self.manifest = create_split(self.root / "source", self.data, independent_images=True)
        self.output = self.root / "measurement output"
        self.seeds = (17, 29)
        self.serial = 0

    def prepare(self):
        return experiment.prepare(
            self.data,
            self.output,
            seeds=self.seeds,
            epochs=2,
            batch_size=4,
            bootstrap_repetitions=20,
        )

    def predictions(self, records, root, manifest, split, *, batch_size):
        rows = [row for row in manifest["rows"] if row["split"] == split]
        scores, identities = {}, {}
        for index, (_, metadata, identity) in enumerate(records):
            pair = (0.2, 0.4) if metadata["seed"] == 17 else (0.6, 0.8)
            name = f"model_{index + 1}"
            scores[name] = [pair[row["label"]] for row in rows]
            identities[name] = identity
        return rows, scores, identities

    def allocate(self, phase, seed):
        self.serial += 1
        folder = self.output / "attempts" / f"{phase}-{seed}-fixture-{self.serial}"
        folder.mkdir(parents=True)
        return folder

    def fit_files(self, folder, seed):
        (folder / "model.keras").write_bytes(f"synthetic model bytes for seed {seed}".encode())
        save_metadata(
            folder / "model.keras",
            self.manifest,
            size=(224, 224),
            architecture="alexnet",
            seed=seed,
            extra={
                "epochs_requested": 2,
                "epochs_completed": 2,
                "batch_size": 4,
                "pretrained": False,
            },
        )
        write_report({"loss": [0.7, 0.6], "val_loss": [0.65, 0.55]}, folder / "model.history.json")

    def bundle(self, phase, seed, folder=None):
        folder = self.allocate(phase, seed) if folder is None else folder
        with patch.object(evaluate, "_predict_models", side_effect=self.predictions):
            if phase == "fit":
                self.fit_files(folder, seed)
                artifact = select_models([folder / "model.keras"], self.data, batch_size=4)
                write_report(artifact, folder / "selection.json")
                files = experiment.FIT_FILES
            else:
                _, fitted = experiment._checkpoint(self.output, "fit", seed)
                report = evaluate.evaluate_models(
                    [fitted / "model.keras"],
                    self.data,
                    batch_size=4,
                    selection=fitted / "selection.json",
                )
                write_report(report, folder / "test-report.json")
                files = ("test-report.json",)
        entry = {
            "phase": phase,
            "seed": seed,
            "plan_sha256": digest(self.output / "plan.json"),
            "directory": folder.relative_to(self.output).as_posix(),
            "files": {name: digest(folder / name) for name in files},
            "details": {"cpu_intra_threads": 2, "cpu_inter_threads": 1, "visible_devices": ["CPU"]},
        }
        write_report(entry, folder / "bundle.json")
        return entry, folder

    def commit(self, phase, seed):
        entry, folder = self.bundle(phase, seed)
        write_report(entry, self.output / f"{phase}-{seed}.json")
        return entry, folder

    def completed_artifacts(self):
        for seed in self.seeds:
            self.commit("fit", seed)
        experiment._lock(self.output, self.seeds)
        for seed in self.seeds:
            self.commit("test", seed)

    def test_prepare_freezes_defaults_without_tensorflow_or_launching(self):
        with (
            patch.dict(sys.modules, {"tensorflow": None}),
            patch.object(experiment, "_launch") as launch,
        ):
            plan = experiment.prepare(self.data, self.output)
            _, loaded, data, manifest = experiment.load_plan(self.output)
        self.assertEqual(plan, loaded)
        self.assertEqual(self.data, data)
        self.assertEqual(self.manifest, manifest)
        self.assertEqual(list(experiment.DEFAULT_SEEDS), plan["recipe"]["seeds"])
        self.assertEqual(
            (30, 16, [224, 224], False),
            tuple(
                plan["recipe"][key]
                for key in ("max_epochs", "batch_size", "image_size", "pretrained")
            ),
        )
        self.assertEqual(
            {"plan.json", "plan.sha256"}, {path.name for path in self.output.iterdir()}
        )
        launch.assert_not_called()
        with self.assertRaises(FileExistsError):
            experiment.prepare(self.data, self.output)

    def test_prepare_rejects_invalid_recipe_and_dataset_overlap(self):
        for kwargs in (
            {"seeds": [1]},
            {"seeds": [1, 1]},
            {"seeds": [True, 2]},
            {"seeds": [-1, 2]},
            {"epochs": 0},
            {"batch_size": True},
            {"bootstrap_repetitions": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                experiment.prepare(self.data, self.output, **kwargs)
            self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError):
            experiment.prepare(self.data, self.data / "nested")

    def test_plan_code_runtime_and_dataset_changes_are_rejected(self):
        self.prepare()
        with patch.object(experiment, "_source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "Code or runtime"):
                experiment.load_plan(self.output)
        with patch.object(experiment, "_versions", return_value={}):
            with self.assertRaisesRegex(ValueError, "Code or runtime"):
                experiment.load_plan(self.output)
        plan_path = self.output / "plan.json"
        original = plan_path.read_bytes()
        plan_path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Plan changed"):
            experiment.load_plan(self.output)
        plan_path.write_bytes(original)
        image = self.data / self.manifest["rows"][0]["path"]
        image.write_bytes(b"altered synthetic image")
        with self.assertRaises(ValueError):
            experiment.load_plan(self.output)

    def test_run_fits_every_seed_before_lock_and_test_and_is_idempotent(self):
        self.prepare()
        calls = []

        def launch(output, phase, seed):
            calls.append((phase, seed))
            if phase == "fit":
                self.assertFalse((output / "evaluation-lock.json").exists())
            else:
                self.assertTrue((output / "evaluation-lock.json").is_file())
                for expected in self.seeds:
                    experiment._checkpoint(output, "fit", expected)
            self.commit(phase, seed)

        with patch.object(experiment, "_launch", side_effect=launch):
            result = experiment.run(self.output)
        self.assertEqual([("fit", 17), ("fit", 29), ("test", 17), ("test", 29)], calls)
        self.assertEqual([17, 29], [row["seed"] for row in result["per_seed"]])
        self.assertEqual(2, result["across_seeds"]["balanced_accuracy"]["count"])
        with patch.object(experiment, "_launch") as unexpected:
            self.assertEqual(result, experiment.run(self.output))
        unexpected.assert_not_called()

    def test_failed_fit_stops_before_test_and_resume_skips_verified_seed(self):
        self.prepare()
        failed_folder = None
        calls = []

        def fail_second(output, phase, seed):
            nonlocal failed_folder
            calls.append((phase, seed))
            if seed == 29:
                failed_folder = self.allocate(phase, seed)
                (failed_folder / "partial.txt").write_text(
                    "preserve failed attempt", encoding="utf-8"
                )
                raise RuntimeError("synthetic worker failure")
            self.commit(phase, seed)

        with patch.object(experiment, "_launch", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic worker failure"):
                experiment.run(self.output)
        self.assertEqual([("fit", 17), ("fit", 29)], calls)
        self.assertFalse((self.output / "evaluation-lock.json").exists())
        self.assertFalse((self.output / "fit-29.json").exists())
        resumed = []

        def succeed(output, phase, seed):
            resumed.append((phase, seed))
            self.commit(phase, seed)

        with patch.object(experiment, "_launch", side_effect=succeed):
            result = experiment.run(self.output)
        self.assertEqual([("fit", 29), ("test", 17), ("test", 29)], resumed)
        self.assertEqual(
            "preserve failed attempt", (failed_folder / "partial.txt").read_text(encoding="utf-8")
        )
        self.assertEqual(5, len(result["experiment"]["attempts"]))

    def test_modified_checkpoint_artifact_stops_resume_without_relaunch(self):
        self.prepare()
        _, folder = self.commit("fit", 17)
        (folder / "model.keras").write_bytes(b"changed synthetic model")
        with patch.object(experiment, "_launch") as launch:
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                experiment.run(self.output)
        launch.assert_not_called()

    def test_fit_receipt_rejects_wrong_seed_even_with_current_file_checksum(self):
        self.prepare()
        entry, folder = self.bundle("fit", 17)
        path = folder / "model.metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        for value in (29, True, "17", None):
            with self.subTest(seed=value):
                metadata["seed"] = value
                path.write_text(json.dumps(metadata), encoding="utf-8")
                entry["files"]["model.metadata.json"] = digest(path)
                with self.assertRaisesRegex(ValueError, "planned seed"):
                    experiment._verify_entry(self.output, entry, "fit", 17)

    def test_launch_commits_only_exit_zero_and_preserves_failed_attempt(self):
        self.prepare()
        commands = []
        return_codes = iter((7, 0))

        def popen(command, **kwargs):
            commands.append((command, kwargs))
            folder = self.output / command[-1]
            self.bundle("fit", 17, folder)
            return SimpleNamespace(
                stdout=io.StringIO("synthetic worker output\n"), wait=lambda: next(return_codes)
            )

        with patch.object(experiment.subprocess, "Popen", side_effect=popen):
            with self.assertRaises(RuntimeError):
                experiment._launch(self.output, "fit", 17)
            failed = next((self.output / "attempts").iterdir())
            preserved = {path.name: path.read_bytes() for path in failed.iterdir()}
            self.assertFalse((self.output / "fit-17.json").exists())
            experiment._launch(self.output, "fit", 17)
        self.assertEqual(2, len(list((self.output / "attempts").iterdir())))
        self.assertEqual(preserved, {path.name: path.read_bytes() for path in failed.iterdir()})
        _, successful = experiment._checkpoint(self.output, "fit", 17)
        self.assertNotEqual(failed, successful)
        self.assertEqual(0, json.loads((successful / "process.json").read_text())["returncode"])
        for command, kwargs in commands:
            self.assertEqual(
                [sys.executable, "-B", "-m", "thyroid_poc.experiment", "worker"], command[:5]
            )
            self.assertEqual([str(self.output), "fit", "17"], command[5:8])
            self.assertNotIn("PYTHONPATH", kwargs["env"])
            self.assertEqual("-1", kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            self.assertEqual("17", kwargs["env"]["PYTHONHASHSEED"])
            self.assertEqual("2", kwargs["env"]["TF_NUM_INTRAOP_THREADS"])
            self.assertEqual("1", kwargs["env"]["TF_NUM_INTEROP_THREADS"])

    def test_launch_rejects_bundle_pointing_at_another_attempt(self):
        self.prepare()
        foreign, _ = self.bundle("fit", 17)

        def popen(command, **kwargs):
            folder = self.output / command[-1]
            write_report(foreign, folder / "bundle.json")
            return SimpleNamespace(stdout=io.StringIO("foreign bundle\n"), wait=lambda: 0)

        with patch.object(experiment.subprocess, "Popen", side_effect=popen):
            with self.assertRaises(ValueError):
                experiment._launch(self.output, "fit", 17)
        self.assertFalse((self.output / "fit-17.json").exists())

    def test_worker_phase_and_environment_guards_precede_tensorflow(self):
        self.prepare()
        folder = self.allocate("fit", 17)
        relative = folder.relative_to(self.output).as_posix()
        with patch.dict(sys.modules, {"tensorflow": None}):
            for phase, seed, directory in (
                ("test", 17, relative),
                ("fit", 99, relative),
                ("fit", 17, "../outside"),
            ):
                with self.subTest(phase=phase, seed=seed), self.assertRaises(ValueError):
                    experiment.worker(self.output, phase, seed, directory)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaisesRegex(ValueError, "environment"):
                    experiment.worker(self.output, "fit", 17, relative)

    def test_worker_fit_writes_bundle_not_committed_checkpoint(self):
        self.prepare()
        folder = self.allocate("fit", 17)
        threading = SimpleNamespace(
            set_intra_op_parallelism_threads=Mock(),
            set_inter_op_parallelism_threads=Mock(),
            get_intra_op_parallelism_threads=lambda: 2,
            get_inter_op_parallelism_threads=lambda: 1,
        )
        fake_tf = SimpleNamespace(
            config=SimpleNamespace(
                threading=threading,
                get_visible_devices=lambda kind=None: (
                    [] if kind == "GPU" else [SimpleNamespace(device_type="CPU")]
                ),
            ),
            keras=SimpleNamespace(backend=SimpleNamespace(clear_session=Mock())),
        )

        def fake_train(root, model_path, **kwargs):
            self.assertEqual(self.data, root)
            self.assertEqual(
                {
                    "architecture": "alexnet",
                    "epochs": 2,
                    "batch_size": 4,
                    "seed": 17,
                    "pretrained": False,
                    "size": (224, 224),
                },
                kwargs,
            )
            self.fit_files(model_path.parent, kwargs["seed"])

        with (
            patch.dict(sys.modules, {"tensorflow": fake_tf}),
            patch.dict(os.environ, experiment._worker_environment(17), clear=True),
            patch("thyroid_poc.train.train_model", side_effect=fake_train),
            patch.object(evaluate, "_predict_models", side_effect=self.predictions),
        ):
            experiment.worker(self.output, "fit", 17, folder.relative_to(self.output).as_posix())
        self.assertFalse((self.output / "fit-17.json").exists())
        entry = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
        experiment._verify_entry(self.output, entry, "fit", 17)
        self.assertEqual(2, entry["details"]["best_validation_loss_epoch"])
        threading.set_intra_op_parallelism_threads.assert_called_once_with(2)
        threading.set_inter_op_parallelism_threads.assert_called_once_with(1)

    def test_incomplete_experiment_cannot_publish_summary(self):
        self.prepare()
        with self.assertRaises(ValueError):
            experiment.summarize(self.output)
        for seed in self.seeds:
            self.commit("fit", seed)
        experiment._lock(self.output, self.seeds)
        self.commit("test", 17)
        with self.assertRaises(FileNotFoundError):
            experiment.summarize(self.output)
        self.assertFalse((self.output / "results.json").exists())

    def test_summary_checks_model_metadata_manifest_and_selection_bindings(self):
        self.prepare()
        self.completed_artifacts()
        entry, folder = experiment._checkpoint(self.output, "test", 17)
        path = folder / "test-report.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        changes = (
            lambda report: report["models"]["model_1"].update(sha256="0" * 64),
            lambda report: report["models"]["model_1"].update(metadata_sha256="0" * 64),
            lambda report: report.update(manifest_sha256="0" * 64),
            lambda report: report["threshold_provenance"].update(artifact_sha256="0" * 64),
            lambda report: report["metrics"]["model_1"].update(threshold=0.99),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                report = copy.deepcopy(original)
                change(report)
                path.write_text(json.dumps(report), encoding="utf-8")
                changed_entry = copy.deepcopy(entry)
                changed_entry["files"]["test-report.json"] = digest(path)
                (self.output / "test-17.json").write_text(
                    json.dumps(changed_entry), encoding="utf-8"
                )
                with self.assertRaises(ValueError):
                    experiment.summarize(self.output)
                self.assertFalse((self.output / "results.json").exists())

    def test_installed_cli_prepare_and_incomplete_summary_from_other_cwd(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "thyroid_poc.experiment",
                "prepare",
                str(self.data),
                str(self.output),
                "--seeds",
                "17",
                "29",
                "--bootstrap-repetitions",
                "20",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("no model fitted", result.stdout)
        self.assertEqual(
            {"plan.json", "plan.sha256"}, {path.name for path in self.output.iterdir()}
        )
        failed = subprocess.run(
            [sys.executable, "-m", "thyroid_poc.experiment", "summarize", str(self.output)],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(0, failed.returncode)
        self.assertIn("incomplete", failed.stderr)
        self.assertFalse((self.output / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
