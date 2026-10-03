"""Synthetic integration and failure-boundary tests; no medical data is used."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_ddti import case, jpeg

from thyroid_poc.data import create_split, read_groups, validate_manifest
from thyroid_poc.ddti import cohort_fingerprint, import_ddti


class DataBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ddti boundaries ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw source with spaces"
        self.raw.mkdir()
        for index in range(12):
            case(
                self.raw,
                100 + index,
                "3" if index < 6 else "4b",
                {1: jpeg((20 + index * 16, 30, 90))},
            )
        # The two labelled images differ. This missing-label case joins their
        # groups and must survive exclusion as a grouping constraint.
        case(
            self.raw,
            9,
            "",
            {1: (self.raw / "100_1.jpg").read_bytes(), 2: (self.raw / "106_1.jpg").read_bytes()},
        )
        self.source_manifest = self.root / "source manifest.json"
        self.source_manifest.write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "path": path.name,
                            "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                        for path in sorted(self.raw.iterdir())
                    ]
                }
            ),
            encoding="utf-8",
        )

    def import_cohort(self):
        destination = self.root / "cohort with spaces"
        manifest = import_ddti(self.raw, destination, expected_source_manifest=self.source_manifest)
        return destination, manifest

    def split_cohort(self, cohort, name="split"):
        return create_split(
            cohort / "data",
            self.root / name,
            groups=read_groups(cohort / "groups.csv"),
            cohort_manifest=cohort / "cohort.json",
            seed=20261002,
        )

    def assert_no_stage(self):
        self.assertEqual([], list(self.root.glob(".ddti-import-*")))
        self.assertEqual([], list(self.root.glob(".split-*")))

    def run_cli(self, module, *args):
        return subprocess.run(
            [sys.executable, "-m", module, *map(str, args)],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_cli_import_then_split_preserves_provenance_and_excluded_bridge(self):
        original = {path.name: path.read_bytes() for path in self.raw.iterdir()}
        cohort, split = self.root / "cohort with spaces", self.root / "split with spaces"
        imported = self.run_cli(
            "thyroid_poc.ddti", self.raw, cohort, "--source-manifest", self.source_manifest
        )
        self.assertEqual(0, imported.returncode, imported.stderr)
        self.assertEqual(12, json.loads(imported.stdout)["counts"]["included_images"])
        result = self.run_cli(
            "thyroid_poc.data",
            cohort / "data",
            split,
            "--groups",
            cohort / "groups.csv",
            "--cohort-manifest",
            cohort / "cohort.json",
            "--seed",
            "20261002",
        )
        self.assertEqual(0, result.returncode, result.stderr)
        manifest = validate_manifest(split)
        self.assertEqual(
            manifest["source_fingerprint"], json.loads(result.stdout)["source_fingerprint"]
        )
        provenance = json.loads((cohort / "cohort.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance, manifest["cohort_provenance"])
        self.assertEqual(2, provenance["counts"]["excluded_images"])
        bridge = [row for row in manifest["rows"] if row["group"] == "ddti-case-9"]
        self.assertEqual({0, 1}, {row["label"] for row in bridge})
        self.assertEqual(1, len({row["split"] for row in bridge}))
        self.assertEqual(
            {
                ("train", 0),
                ("train", 1),
                ("validation", 0),
                ("validation", 1),
                ("test", 0),
                ("test", 1),
            },
            {(row["split"], row["label"]) for row in manifest["rows"]},
        )
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.raw.iterdir()})
        self.assert_no_stage()

    def test_cli_conflicting_group_options_fail_before_output(self):
        cohort, _ = self.import_cohort()
        destination = self.root / "never-created"
        result = self.run_cli(
            "thyroid_poc.data",
            cohort / "data",
            destination,
            "--groups",
            cohort / "groups.csv",
            "--independent-images",
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("not allowed with argument", result.stderr)
        self.assertFalse(destination.exists())
        self.assert_no_stage()

    def test_import_late_destination_collision_preserves_competing_output(self):
        destination = self.root / "cohort"
        real_write = Path.write_bytes

        def collide(path, data):
            result = real_write(path, data)
            if path.suffix == ".jpg" and ".ddti-import-" in str(path) and not destination.exists():
                destination.mkdir()
                (destination / "sentinel.txt").write_text("another writer", encoding="utf-8")
            return result

        with patch.object(Path, "write_bytes", collide):
            with self.assertRaises(FileExistsError):
                import_ddti(self.raw, destination)
        self.assertEqual(["sentinel.txt"], [path.name for path in destination.iterdir()])
        self.assertEqual(
            "another writer", (destination / "sentinel.txt").read_text(encoding="utf-8")
        )
        self.assert_no_stage()

    def test_split_late_destination_collision_preserves_competing_output(self):
        cohort, _ = self.import_cohort()
        destination = self.root / "split"
        real_copy = shutil.copyfile

        def collide(source, target, *args, **kwargs):
            result = real_copy(source, target, *args, **kwargs)
            if not destination.exists():
                destination.mkdir()
                (destination / "sentinel.txt").write_text("another writer", encoding="utf-8")
            return result

        with patch("thyroid_poc.data.shutil.copyfile", collide):
            with self.assertRaises(FileExistsError):
                self.split_cohort(cohort)
        self.assertEqual(["sentinel.txt"], [path.name for path in destination.iterdir()])
        self.assertEqual(
            "another writer", (destination / "sentinel.txt").read_text(encoding="utf-8")
        )
        self.assert_no_stage()

    def test_source_manifest_mutation_during_import_rejects_atomic_publish(self):
        real_write = Path.write_bytes
        changed = False

        def mutate(path, data):
            nonlocal changed
            result = real_write(path, data)
            if path.suffix == ".jpg" and ".ddti-import-" in str(path) and not changed:
                changed = True
                content = self.source_manifest.read_text(encoding="utf-8")
                self.source_manifest.write_text(content + "\n", encoding="utf-8")
            return result

        with patch.object(Path, "write_bytes", mutate):
            with self.assertRaisesRegex(ValueError, "source manifest changed"):
                self.import_cohort()
        self.assertFalse((self.root / "cohort with spaces").exists())
        self.assert_no_stage()

    def test_group_csv_rejects_extra_missing_and_empty_path_fields(self):
        csv_path = self.root / "groups.csv"
        for row in ("image.jpg,group,unquoted extra field", "image.jpg", ",group"):
            with self.subTest(row=row):
                csv_path.write_text("path,group\n" + row + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_groups(csv_path)

    def test_recomputed_fingerprint_cannot_bind_contradictory_tirads_target(self):
        cohort, manifest = self.import_cohort()
        for value in ("5", "", "unsupported"):
            with self.subTest(tirads=value):
                changed = copy.deepcopy(manifest)
                row = next(row for row in changed["rows"] if row["label"] == 0)
                row["tirads"] = value  # Still included in the lower TIRADS class.
                changed["cohort_fingerprint"] = cohort_fingerprint(changed)
                (cohort / "cohort.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.split_cohort(cohort)
                self.assertFalse((self.root / "split").exists())
                self.assert_no_stage()


if __name__ == "__main__":
    unittest.main()
