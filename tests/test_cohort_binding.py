import copy
import json
import tempfile
import unittest
from pathlib import Path

from test_ddti import case, jpeg

from thyroid_poc.data import create_split, read_groups, validate_manifest
from thyroid_poc.ddti import cohort_fingerprint, import_ddti


class CohortBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = self.root / "raw"
        raw.mkdir()
        for index in range(12):
            case(raw, 100 + index, "3" if index < 6 else "4b", {1: jpeg((index * 15, 30, 90))})
        self.cohort = import_ddti(raw, self.root / "cohort")
        self.cohort_path = self.root / "cohort/cohort.json"

    def split(self):
        return create_split(
            self.root / "cohort/data",
            self.root / "split",
            groups=read_groups(self.root / "cohort/groups.csv"),
            cohort_manifest=self.cohort_path,
        )

    def test_import_provenance_survives_split_and_revalidation(self):
        manifest = self.split()
        self.assertEqual(self.cohort, manifest["cohort_provenance"])
        self.assertEqual(manifest, validate_manifest(self.root / "split"))
        tampered = copy.deepcopy(manifest)
        tampered["cohort_provenance"]["source_files"][0]["sha256"] = "changed"
        (self.root / "split/manifest.json").write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cohort provenance"):
            validate_manifest(self.root / "split")

    def test_self_consistent_but_mismatched_cohort_rejected(self):
        for field, value in (("group", "wrong"), ("label", 1), ("sha256", "wrong"), ("size", 1)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.cohort)
                changed["rows"][0][field] = value
                changed["cohort_fingerprint"] = cohort_fingerprint(changed)
                self.cohort_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "do not match"):
                    self.split()
                self.assertFalse((self.root / "split").exists())


if __name__ == "__main__":
    unittest.main()
