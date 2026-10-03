import csv
import hashlib
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from thyroid_poc.ddti import CLASS_NAMES, import_ddti


def jpeg(color, *, comment=None):
    buffer = io.BytesIO()
    Image.new("RGB", (9, 7), color).save(buffer, format="JPEG", quality=100, subsampling=0)
    raw = buffer.getvalue()
    if comment is not None:
        payload = comment.encode("ascii")
        raw = raw[:2] + b"\xff\xfe" + (len(payload) + 2).to_bytes(2, "big") + payload + raw[2:]
    return raw


def case(source, number, tirads, views):
    root = ET.Element("case")
    ET.SubElement(root, "number").text = str(number)
    ET.SubElement(root, "tirads").text = tirads
    for view, data in views.items():
        mark = ET.SubElement(root, "mark")
        ET.SubElement(mark, "image").text = str(view)
        (source / f"{number}_{view}.jpg").write_bytes(data)
    (source / f"{number}.xml").write_bytes(ET.tostring(root))


class DDTIImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        case(self.source, 12, "2", {1: jpeg((10, 20, 30)), 11: jpeg((30, 20, 10))})
        case(self.source, 123, "4a", {1: jpeg((200, 20, 50))})
        case(self.source, 9, "", {1: jpeg((90, 90, 90))})

    def run_import(self, name="cohort", **kwargs):
        return import_ddti(self.source, self.root / name, **kwargs)

    def assert_no_stage(self):
        self.assertEqual([], list(self.root.glob(".ddti-import-*")))

    def test_full_case_view_ids_target_mapping_and_metadata_layout(self):
        manifest = self.run_import()
        rows = {row["source_image"]: row for row in manifest["rows"]}
        self.assertEqual("12", rows["12_11.jpg"]["case"])
        self.assertEqual("11", rows["12_11.jpg"]["view"])
        self.assertEqual(rows["12_1.jpg"]["group"], rows["12_11.jpg"]["group"])
        self.assertNotEqual(rows["12_1.jpg"]["group"], rows["123_1.jpg"]["group"])
        self.assertEqual(0, rows["12_1.jpg"]["label"])
        self.assertEqual(1, rows["123_1.jpg"]["label"])
        self.assertEqual(list(CLASS_NAMES), manifest["target_schema"]["class_names"])
        self.assertEqual(["missing_tirads"], [row["reason"] for row in manifest["excluded"]])
        self.assertEqual(set(CLASS_NAMES), {p.name for p in (self.root / "cohort/data").iterdir()})
        with (self.root / "cohort/groups.csv").open(newline="", encoding="utf-8") as stream:
            groups = list(csv.DictReader(stream))
        self.assertEqual({row["path"] for row in manifest["rows"]}, {row["path"] for row in groups})
        self.assertTrue(all((self.root / "cohort/data" / row["path"]).is_file() for row in groups))

    def test_every_supported_tirads_value_and_missing_policy(self):
        for index, tirads in enumerate(("2", "3", "4a", "4b", "4c", "5"), start=200):
            case(self.source, index, tirads, {1: jpeg(((index - 199) * 30, 110, 7))})
        manifest = self.run_import()
        labels = {row["tirads"]: row["label"] for row in manifest["rows"]}
        self.assertEqual({"2": 0, "3": 0, "4a": 1, "4b": 1, "4c": 1, "5": 1}, labels)

    def test_unsupported_tirads_rejected_without_output(self):
        for value in ("4", "4A", "unknown", "6", "malignant"):
            with self.subTest(value=value):
                case(self.source, 9, value, {1: jpeg((90, 90, 90))})
                with self.assertRaisesRegex(ValueError, "Unsupported TIRADS"):
                    self.run_import()
                self.assertFalse((self.root / "cohort").exists())
                self.assert_no_stage()

    def test_missing_case_bridges_included_cases_before_exclusion(self):
        a, b = jpeg((10, 20, 30)), jpeg((30, 40, 50))
        case(self.source, 200, "3", {1: b})
        case(self.source, 9, "", {1: a, 2: b})
        manifest = self.run_import()
        rows = {row["case"]: row for row in manifest["rows"]}
        self.assertEqual(rows["12"]["group"], rows["200"]["group"])
        self.assertEqual("ddti-case-9", rows["12"]["group"])
        self.assertEqual(2, len(manifest["excluded"]))

    def test_missing_case_bridge_keeps_different_label_images_in_same_component(self):
        case(
            self.source,
            9,
            "",
            {
                1: (self.source / "12_1.jpg").read_bytes(),
                2: (self.source / "123_1.jpg").read_bytes(),
            },
        )
        manifest = self.run_import()
        self.assertEqual(1, len({row["group"] for row in manifest["rows"]}))
        self.assertEqual({0, 1}, {row["label"] for row in manifest["rows"]})
        self.assertEqual(["9", "12", "123"], manifest["components"][0]["cases"])

    def test_exact_image_conflicting_binary_labels_rejected(self):
        case(self.source, 200, "5", {1: (self.source / "12_1.jpg").read_bytes()})
        with self.assertRaisesRegex(ValueError, "conflicting included binary labels"):
            self.run_import()
        self.assertFalse((self.root / "cohort").exists())
        self.assert_no_stage()

    def test_reencoded_metadata_equal_pixels_connect_cases(self):
        original = (self.source / "12_1.jpg").read_bytes()
        variant = jpeg((10, 20, 30), comment="different metadata, identical decoded pixels")
        self.assertNotEqual(original, variant)
        case(self.source, 200, "3", {1: variant})
        manifest = self.run_import()
        rows = {row["case"]: row for row in manifest["rows"] if row["view"] == "1"}
        self.assertEqual(rows["12"]["rgb_sha256"], rows["200"]["rgb_sha256"])
        self.assertEqual(rows["12"]["group"], rows["200"]["group"])

    def test_decoded_pixel_conflicting_labels_rejected(self):
        case(self.source, 200, "5", {1: jpeg((10, 20, 30), comment="new metadata")})
        with self.assertRaisesRegex(ValueError, "conflicting included binary labels"):
            self.run_import()

    def test_deterministic_replay_and_original_bytes_preserved(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        a, b = self.run_import("a"), self.run_import("b")
        self.assertEqual(a, b)
        self.assertEqual(a, json.loads((self.root / "a/cohort.json").read_text(encoding="utf-8")))
        for file in ("cohort.json", "groups.csv"):
            self.assertEqual(
                (self.root / "a" / file).read_bytes(), (self.root / "b" / file).read_bytes()
            )
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        for row in a["rows"]:
            self.assertEqual(
                before[row["source_image"]], (self.root / "a/data" / row["path"]).read_bytes()
            )
            self.assertEqual(
                hashlib.sha256(before[row["source_xml"]]).hexdigest(), row["xml_sha256"]
            )
        self.assert_no_stage()

    def test_invalid_case_number_marks_missing_extra_and_malformed_xml(self):
        path = self.source / "12.xml"
        original = path.read_bytes()
        variants = [
            original.replace(b"<number>12</number>", b"<number>1</number>"),
            original.replace(b"<image>11</image>", b"<image>1</image>"),
            original.replace(b"<image>11</image>", b"<image>../123_1</image>"),
            original.replace(b"<image>11</image>", b"<image>99</image>"),
            original.replace(b"<tirads>2</tirads>", b"<tirads>2</tirads><tirads>3</tirads>"),
            b"not XML",
            b'<!DOCTYPE case [<!ENTITY x "12">]><case><number>&x;</number></case>',
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                path.write_bytes(variant)
                with self.assertRaises(ValueError):
                    self.run_import()
                self.assert_no_stage()
        path.write_bytes(original)
        (self.source / "777_1.jpg").write_bytes(jpeg((1, 2, 3)))
        with self.assertRaisesRegex(ValueError, "agree exactly"):
            self.run_import()

    def test_invalid_image_or_nested_source_rejected(self):
        image = self.source / "9_1.jpg"
        original = image.read_bytes()
        image.write_bytes(b"invalid JPEG")
        with self.assertRaisesRegex(ValueError, "decoded"):
            self.run_import()
        image.write_bytes(original)
        (self.source / "nested").mkdir()
        with self.assertRaisesRegex(ValueError, "nested"):
            self.run_import()

    def test_utf16_entity_declaration_cannot_bypass_xml_restriction(self):
        text = '<!DOCTYPE case [<!ENTITY number "12">]><case><number>&number;</number><tirads>2</tirads><mark><image>1</image></mark><mark><image>11</image></mark></case>'
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            with self.subTest(encoding=encoding):
                (self.source / "12.xml").write_bytes(text.encode(encoding))
                with self.assertRaisesRegex(ValueError, "UTF-8"):
                    self.run_import()
                self.assertFalse((self.root / "cohort").exists())

    def test_expected_source_manifest_covers_exact_preserved_files(self):
        path = self.root / "source-manifest.json"
        records = [
            {
                "path": p.name,
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(self.source.iterdir())
        ]
        path.write_text(
            json.dumps({"files": records, "archive_sha256": "archive-provenance"}), encoding="utf-8"
        )
        manifest = self.run_import("ok", expected_source_manifest=path)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), manifest["source_binding"]["sha256"]
        )
        (self.source / "9_1.jpg").write_bytes(jpeg((88, 99, 111)))
        with self.assertRaisesRegex(ValueError, "expected source manifest"):
            self.run_import("bad", expected_source_manifest=path)

    def test_existing_nested_and_nonexisting_parent_destinations_rejected(self):
        self.run_import()
        with self.assertRaises(FileExistsError):
            self.run_import()
        with self.assertRaisesRegex(ValueError, "contain each other"):
            import_ddti(self.source, self.source / "nested-output")
        with self.assertRaisesRegex(ValueError, "parent must already exist"):
            import_ddti(self.source, self.root / "missing-parent/cohort")

    def test_source_change_even_excluded_xml_detected_and_stage_cleaned(self):
        real_write = Path.write_bytes
        changed = False

        def mutate_when_copying(path, data):
            nonlocal changed
            result = real_write(path, data)
            if path.suffix == ".jpg" and ".ddti-import-" in str(path) and not changed:
                changed = True
                xml = self.source / "9.xml"
                real_write(xml, xml.read_bytes() + b"\n")
            return result

        with patch.object(Path, "write_bytes", mutate_when_copying):
            with self.assertRaisesRegex(ValueError, "Source bytes changed"):
                self.run_import()
        self.assertFalse((self.root / "cohort").exists())
        self.assert_no_stage()

    def test_symlink_source_entry_and_output_parent_rejected(self):
        link = self.root / "linked-source"
        try:
            link.symlink_to(self.source, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("OS does not permit symlink creation for this process")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            import_ddti(link, self.root / "cohort")
        output_parent = self.root / "linked-parent"
        output_parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            import_ddti(self.source, output_parent / "cohort")
        target = self.source / "9_1.jpg"
        outside = self.root / "outside.jpg"
        target.rename(outside)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            self.run_import()

    def test_symlink_guard_runs_before_reading_source_entry(self):
        target = self.source / "9_1.jpg"
        original = Path.is_symlink

        def reports_symlink(path):
            return path == target or original(path)

        with patch.object(Path, "is_symlink", reports_symlink):
            with self.assertRaisesRegex(ValueError, "Symlinks"):
                self.run_import()
        self.assertFalse((self.root / "cohort").exists())


if __name__ == "__main__":
    unittest.main()
