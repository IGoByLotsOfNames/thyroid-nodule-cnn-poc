"""Opt-in CPU experiment integration on synthetic colours, never medical data."""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_selection import image_fixture

from thyroid_poc.artifacts import load_metadata
from thyroid_poc.data import create_split, digest, validate_manifest
from thyroid_poc.experiment import FIT_FILES


@unittest.skipUnless(
    os.environ.get("RUN_ML_TESTS") == "1",
    "Set RUN_ML_TESTS=1 for actual synthetic experiment worker processes",
)
class ExperimentRuntimeTests(unittest.TestCase):
    def cli(self, cwd, *arguments):
        result = subprocess.run(
            [sys.executable, "-m", "thyroid_poc.experiment", *map(str, arguments)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600,  # Hang guard only; no expected run duration is asserted.
            check=False,
        )
        self.assertEqual(
            0,
            result.returncode,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result

    def test_installed_runner_two_seed_cpu_lifecycle_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory(prefix="synthetic experiment runtime ") as temporary:
            root = Path(temporary)
            source, data, output = root / "colour source", root / "split data", root / "run output"
            image_fixture(source)
            manifest = create_split(source, data, independent_images=True, seed=20261002)
            source_before = {
                path.relative_to(source).as_posix(): digest(path)
                for path in source.rglob("*")
                if path.is_file()
            }
            manifest_hash = digest(data / "manifest.json")
            seeds = (17, 29)
            self.cli(
                root,
                "prepare",
                data,
                output,
                "--seeds",
                *seeds,
                "--epochs",
                1,
                "--batch-size",
                4,
                "--bootstrap-repetitions",
                20,
            )
            self.assertEqual({"plan.json", "plan.sha256"}, {path.name for path in output.iterdir()})
            plan_hash = digest(output / "plan.json")
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual([17, 29], plan["recipe"]["seeds"])
            self.assertFalse(plan["recipe"]["pretrained"])

            completed = self.cli(root, "run", output)
            # Verify phase order without assumptions about training speed or score.
            events = [f"fit: seed {seed}" for seed in seeds] + [
                f"test: seed {seed}" for seed in seeds
            ]
            positions = [completed.stdout.index(event) for event in events]
            self.assertEqual(sorted(positions), positions)
            lock = json.loads((output / "evaluation-lock.json").read_text(encoding="utf-8"))
            self.assertEqual(plan_hash, lock["plan_sha256"])
            self.assertEqual({str(seed) for seed in seeds}, set(lock["fit_checkpoints"]))

            for seed in seeds:
                fit_path = output / f"fit-{seed}.json"
                fit = json.loads(fit_path.read_text(encoding="utf-8"))
                fitted = output / fit["directory"]
                self.assertEqual(digest(fit_path), lock["fit_checkpoints"][str(seed)])
                self.assertEqual(set(FIT_FILES), set(fit["files"]))
                self.assertIn("selection.json", fit["files"])
                metadata = load_metadata(fitted / "model.keras")
                self.assertEqual(seed, metadata["seed"])
                self.assertEqual("alexnet", metadata["architecture"])
                self.assertEqual([224, 224], metadata["image_size"])
                self.assertEqual(manifest["class_names"], metadata["class_names"])
                self.assertEqual(1, metadata["training"]["epochs_requested"])
                self.assertEqual(1, metadata["training"]["epochs_completed"])
                self.assertEqual(4, metadata["training"]["batch_size"])
                self.assertFalse(metadata["training"]["pretrained"])
                self.assertEqual(plan["versions"]["tensorflow"], metadata["training"]["tensorflow"])
                self.assertEqual(plan["versions"]["keras"], metadata["training"]["keras"])

                for phase, names in (("fit", set(FIT_FILES)), ("test", {"test-report.json"})):
                    checkpoint = json.loads(
                        (output / f"{phase}-{seed}.json").read_text(encoding="utf-8")
                    )
                    folder = output / checkpoint["directory"]
                    self.assertEqual(
                        (phase, seed, plan_hash),
                        (checkpoint["phase"], checkpoint["seed"], checkpoint["plan_sha256"]),
                    )
                    self.assertEqual(names, set(checkpoint["files"]))
                    self.assertEqual(
                        checkpoint, json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
                    )
                    for name, checksum in checkpoint["files"].items():
                        self.assertEqual(checksum, digest(folder / name))
                    details = checkpoint["details"]
                    self.assertEqual(2, details["cpu_intra_threads"])
                    self.assertEqual(1, details["cpu_inter_threads"])
                    self.assertEqual({"CPU"}, set(details["visible_devices"]))
                    self.assertEqual("-1", details["worker_environment"]["CUDA_VISIBLE_DEVICES"])
                    self.assertEqual(str(seed), details["worker_environment"]["PYTHONHASHSEED"])
                    process = json.loads((folder / "process.json").read_text(encoding="utf-8"))
                    self.assertEqual(0, process["returncode"])
                    self.assertEqual((phase, seed), (process["phase"], process["seed"]))
                    self.assertTrue((folder / "process.log").is_file())

                tested = (
                    output
                    / json.loads((output / f"test-{seed}.json").read_text(encoding="utf-8"))[
                        "directory"
                    ]
                )
                report = json.loads((tested / "test-report.json").read_text(encoding="utf-8"))
                self.assertEqual("test", report["split"])
                self.assertEqual(manifest_hash, report["manifest_sha256"])
                self.assertEqual(
                    digest(fitted / "model.keras"), report["models"]["model_1"]["sha256"]
                )
                self.assertTrue(report["threshold_provenance"]["selection_verified"])
                self.assertEqual(
                    digest(fitted / "selection.json"),
                    report["threshold_provenance"]["artifact_sha256"],
                )

            result_path = output / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(2, result["seed_count"])
            self.assertEqual([17, 29], [entry["seed"] for entry in result["per_seed"]])
            for statistics in result["across_seeds"].values():
                self.assertEqual(2, statistics["count"])
                for name in ("mean", "sample_sd", "sample_variance", "min", "max"):
                    self.assertTrue(math.isfinite(statistics[name]))
            bootstrap = result["group_bootstrap"]
            self.assertEqual("complete", bootstrap["status"])
            self.assertEqual(20, bootstrap["valid_draws"])
            for interval in bootstrap["intervals"].values():
                self.assertTrue(math.isfinite(interval["lower"]))
                self.assertTrue(math.isfinite(interval["upper"]))
                self.assertLessEqual(interval["lower"], interval["upper"])

            attempts_before = {folder.name for folder in (output / "attempts").iterdir()}
            self.assertEqual(4, len(attempts_before))
            frozen_artifacts = {
                path.relative_to(output).as_posix(): digest(path)
                for path in output.rglob("*")
                if path.is_file()
            }
            self.cli(root, "summarize", output)
            self.cli(root, "run", output)
            self.assertEqual(
                attempts_before, {folder.name for folder in (output / "attempts").iterdir()}
            )
            self.assertEqual(
                frozen_artifacts,
                {
                    path.relative_to(output).as_posix(): digest(path)
                    for path in output.rglob("*")
                    if path.is_file()
                },
            )
            self.assertEqual(manifest, validate_manifest(data))
            self.assertEqual(
                source_before,
                {
                    path.relative_to(source).as_posix(): digest(path)
                    for path in source.rglob("*")
                    if path.is_file()
                },
            )


if __name__ == "__main__":
    unittest.main()
