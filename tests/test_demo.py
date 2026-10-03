"""Offline demo contracts; the optional CNN CLI smoke uses synthetic PNGs only."""

import base64
import copy
import hashlib
import html
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("demo", REPO / "demo.py")
demo = importlib.util.module_from_spec(SPEC)
sys.modules["demo"] = demo
SPEC.loader.exec_module(demo)


class Document(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.tags = []
        self.attributes = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.append((tag, dict(attrs)))


class DemoTests(unittest.TestCase):
    def setUp(self):
        original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        self.fixture, self.recorded, self.fixture_hash = demo.load_examples()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def checkout(self):
        root = self.root / "checkout"
        shutil.copytree(REPO / "examples", root / "examples")
        return root

    def payload(self):
        return {
            "schema_version": 1,
            "kind": "thyroid_demo",
            "synthetic": demo.synthetic_walkthrough(self.fixture, self.fixture_hash),
            "recorded_measurement": copy.deepcopy(self.recorded),
            "cnn_smoke": None,
        }

    def cli(self, output, *, cnn=False, isolated=True):
        unrelated = self.root / "unrelated"
        unrelated.mkdir(exist_ok=True)
        command = [sys.executable]
        command += ["-I", "-S"] if isolated else ["-B"]
        command += [str(REPO / "demo.py"), "--output", str(output)]
        if cnn:
            command.append("--cnn")
        return subprocess.run(
            command,
            cwd=unrelated,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=240 if cnn else 60,
            check=False,
        )

    def test_offline_cli_without_site_packages_and_explicit_collision(self):
        output = self.root / "fresh"
        result = self.cli(output)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
        synthetic = payload["synthetic"]
        self.assertEqual(0.7, synthetic["selection"]["threshold"])
        self.assertEqual(0.75, synthetic["selected_metrics"]["balanced_accuracy"])
        self.assertEqual(0.5, synthetic["fixed_metrics"]["balanced_accuracy"])
        self.assertEqual({"train": 8, "validation": 4, "test": 4}, synthetic["counts"])
        self.assertEqual(self.recorded, payload["recorded_measurement"])
        self.assertIsNone(payload["cnn_smoke"])
        self.assertIn("Synthetic probabilities are hand-authored", result.stdout)
        self.assertIn("selected=75.00%; fixed 0.5=50.00%", result.stdout)
        self.assertIn(
            "Recorded TIRADS mean balanced accuracy: 55.79% (five seeds; no rerun)", result.stdout
        )
        source = (output / "report.html").read_text(encoding="utf-8")
        document = Document(source)
        self.assertTrue(source.lower().startswith("<!doctype html>"))
        self.assertIn("75.00%", source)
        self.assertIn("55.79%", source)
        self.assertIn("default-src 'none'", source)
        ids = {attrs.get("id") for _, attrs in document.attributes}
        self.assertTrue({"synthetic", "recorded"}.issubset(ids))
        self.assertFalse({"script", "iframe", "link"}.intersection(document.tags))
        images = [attrs["src"] for tag, attrs in document.attributes if tag == "img"]
        self.assertEqual(4, len(images))
        for uri in images:
            self.assertTrue(uri.startswith("data:image/png;base64,"))
            self.assertTrue(
                base64.b64decode(uri.split(",", 1)[1], validate=True).startswith(
                    b"\x89PNG\r\n\x1a\n"
                )
            )
        self.assertNotIn("@import", source)
        self.assertNotIn("url(", source)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        collision = self.cli(output)
        self.assertNotEqual(0, collision.returncode)
        self.assertIn("Demo stopped:", collision.stderr)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_default_outputs_are_fresh_and_collision_never_overwrites(self):
        root = self.checkout()
        with patch.object(demo, "ROOT", root), patch.object(demo, "datetime") as clock:
            clock.now.side_effect = [
                datetime(2026, 10, 3, 0, 0, 0, 1, timezone.utc),
                datetime(2026, 10, 3, 0, 0, 0, 2, timezone.utc),
                datetime(2026, 10, 3, 0, 0, 0, 1, timezone.utc),
            ]
            first, _ = demo.run_demo()
            before = (first / "report.json").read_bytes()
            second, _ = demo.run_demo()
            self.assertNotEqual(first, second)
            self.assertEqual(root / "demo-output", first.parent)
            with self.assertRaises(FileExistsError):
                demo.run_demo()
            self.assertEqual(before, (first / "report.json").read_bytes())
            for protected in (
                root,
                root / "examples/new",
                root / "src/new",
                root / "demo_support/new",
            ):
                with (
                    self.subTest(protected=protected),
                    self.assertRaisesRegex(ValueError, "outside the source"),
                ):
                    demo.run_demo(protected)
                self.assertFalse((protected / "report.json").exists())

    def test_checksum_tampering_fails_before_output_creation(self):
        root = self.checkout()
        targets = [
            root / "examples/demo-input.json",
            root / "examples/recorded-measurement.json",
            root / "examples/images" / self.fixture["samples"][0]["source"],
        ]
        with patch.object(demo, "ROOT", root):
            for index, target in enumerate(targets):
                original = target.read_bytes()
                target.write_bytes(original + b" ")
                output = self.root / f"tampered-{index}"
                try:
                    with self.subTest(target=target), self.assertRaisesRegex(ValueError, "changed"):
                        demo.run_demo(output)
                    self.assertFalse(output.exists())
                finally:
                    target.write_bytes(original)

    def test_unlisted_image_is_rejected_before_cnn_can_scan_it(self):
        root = self.checkout()
        images = root / "examples/images"
        shutil.copyfile(images / self.fixture["samples"][0]["source"], images / "extra.png")
        with patch.object(demo, "ROOT", root), self.assertRaises(ValueError):
            demo.load_examples()

    def test_synthetic_contract_rejects_semantic_corruption_after_rehash(self):
        root = self.checkout()
        changes = [
            ("probability", lambda f: f["samples"][4].update(probability=float("nan"))),
            ("label", lambda f: f["samples"][0].update(label=True)),
            ("partition overlap", lambda f: f["samples"][4].update(group=f["samples"][0]["group"])),
            ("path traversal", lambda f: f["samples"][0].update(source="../demo-input.json")),
        ]
        with patch.object(demo, "ROOT", root):
            for name, mutate in changes:
                fixture = copy.deepcopy(self.fixture)
                mutate(fixture)
                path = root / "examples/demo-input.json"
                path.write_text(json.dumps(fixture), encoding="utf-8")
                checksums = json.loads(
                    (root / "examples/demo-checksums.json").read_text(encoding="utf-8")
                )
                checksums["demo-input.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
                (root / "examples/demo-checksums.json").write_text(
                    json.dumps(checksums), encoding="utf-8"
                )
                with self.subTest(name=name), self.assertRaises(ValueError):
                    demo.load_examples()

    def test_threshold_selection_uses_validation_only(self):
        original = demo.synthetic_walkthrough(self.fixture, self.fixture_hash)
        altered_test = copy.deepcopy(self.fixture)
        for row in altered_test["samples"]:
            if row["split"] == "test":
                row["label"] = 1 - row["label"]
        test_result = demo.synthetic_walkthrough(altered_test, self.fixture_hash)
        self.assertEqual(original["selection"], test_result["selection"])
        self.assertEqual(0.25, test_result["selected_metrics"]["balanced_accuracy"])
        altered_validation = copy.deepcopy(self.fixture)
        for row in altered_validation["samples"]:
            if row["split"] == "validation":
                row["probability"] = 0.9 if row["label"] else 0.1
        validation_result = demo.synthetic_walkthrough(altered_validation, self.fixture_hash)
        self.assertEqual(0.5, validation_result["selection"]["threshold"])
        self.assertEqual(original["samples"], validation_result["samples"])

    def test_missing_optional_runtime_fails_before_creating_output(self):
        output = self.root / "missing-runtime"
        with (
            patch.object(demo.importlib.util, "find_spec", return_value=None),
            self.assertRaisesRegex(ValueError, "optional runtime"),
        ):
            demo.run_demo(output, cnn=True)
        self.assertFalse(output.exists())

    def test_renderer_escapes_untrusted_text_and_omits_unsafe_images(self):
        from demo_support.report import render_report

        payload = self.payload()
        malicious = '<script>alert("x")</script><img src=x onerror="alert(1)">'
        payload["synthetic"]["notice"] = malicious
        payload["synthetic"]["class_names"][0] = malicious
        payload["synthetic"]["samples"][0].update(source=malicious, group=malicious)
        payload["recorded_measurement"]["interpretation"] = malicious
        payload["recorded_measurement"]["provenance"]["verification"] = malicious
        safe_image = "data:image/png;base64," + base64.b64encode(b"fixture").decode("ascii")
        source = render_report(payload, {malicious: safe_image})
        self.assertIn(html.escape(malicious, quote=True), source)
        parsed = Document(source)
        self.assertNotIn("script", parsed.tags)
        self.assertEqual(1, parsed.tags.count("img"))
        self.assertFalse(
            any(key.startswith("on") for _, attrs in parsed.attributes for key in attrs)
        )
        for unsafe in (
            "javascript:alert(1)",
            "https://example.com/x.png",
            "data:image/svg+xml;base64,PHN2Zz4=",
            'data:image/png;base64,AAAA" onerror="alert(1)',
        ):
            with self.subTest(uri=unsafe):
                rejected = render_report(payload, {malicious: unsafe})
                self.assertNotIn("img", Document(rejected).tags)
                self.assertIn("No image", rejected)

    def test_renderer_optional_sections_and_payload_contract(self):
        from demo_support.report import render_report

        payload = self.payload()
        payload["recorded_measurement"] = None
        payload["cnn_smoke"] = {
            "notice": "Synthetic execution fixture only",
            "metrics": payload["synthetic"]["selected_metrics"],
            "class_names": payload["synthetic"]["class_names"],
            "seed": 17,
            "epochs": 1,
            "threshold": 0.7,
        }
        source = render_report(payload, {})
        self.assertIn("No recorded measurement is bundled.", source)
        self.assertIn("Synthetic execution fixture only", source)
        self.assertNotIn("55.79%", source)
        payload["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            render_report(payload, {})

    @unittest.skipUnless(
        os.environ.get("RUN_ML_TESTS") == "1", "Set RUN_ML_TESTS=1 for the synthetic CNN CLI smoke"
    )
    def test_cnn_cli_saves_bound_model_selection_and_finite_report(self):
        output = self.root / "cnn-output"
        result = self.cli(output, cnn=True, isolated=False)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        payload = json.loads((output / "report.json").read_text(encoding="utf-8"))
        folder = output / "cnn-smoke"
        metadata = json.loads((folder / "model.metadata.json").read_text(encoding="utf-8"))
        selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        report = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
        manifest = json.loads((folder / "split/manifest.json").read_text(encoding="utf-8"))
        model_hash = hashlib.sha256((folder / "model.keras").read_bytes()).hexdigest()
        metadata_hash = hashlib.sha256((folder / "model.metadata.json").read_bytes()).hexdigest()
        self.assertEqual(model_hash, metadata["model_sha256"])
        self.assertEqual(17, metadata["seed"])
        self.assertEqual(1, metadata["training"]["epochs_requested"])
        self.assertEqual(1, metadata["training"]["epochs_completed"])
        self.assertEqual(4, metadata["training"]["batch_size"])
        self.assertFalse(metadata["training"]["pretrained"])
        self.assertEqual(
            [{"sha256": model_hash, "metadata_sha256": metadata_hash}], selection["model_bindings"]
        )
        self.assertEqual(model_hash, report["models"]["model_1"]["sha256"])
        self.assertEqual(metadata_hash, report["models"]["model_1"]["metadata_sha256"])
        self.assertEqual(
            hashlib.sha256((folder / "selection.json").read_bytes()).hexdigest(),
            report["threshold_provenance"]["artifact_sha256"],
        )
        self.assertTrue(report["threshold_provenance"]["selection_verified"])
        self.assertEqual(
            selection["decisions"][model_hash]["threshold"],
            report["metrics"]["model_1"]["threshold"],
        )
        self.assertEqual(16, len(manifest["rows"]))
        self.assertEqual(
            {r["sha256"] for r in manifest["rows"] if r["split"] == "test"},
            {r["sha256"] for r in report["samples"]},
        )
        self.assertEqual(self.recorded, payload["recorded_measurement"])
        self.assertEqual(17, payload["cnn_smoke"]["seed"])
        self.assertEqual(1, payload["cnn_smoke"]["epochs"])
        self.assertIn("completed one epoch", result.stdout)

        def assert_finite(value):
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value))
            elif isinstance(value, dict):
                for item in value.values():
                    assert_finite(item)
            elif isinstance(value, list):
                for item in value:
                    assert_finite(item)

        assert_finite(payload)
        assert_finite(report)
        assert_finite(selection)
        assert_finite(json.loads((folder / "model.history.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
