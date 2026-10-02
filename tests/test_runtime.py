"""CPU-only synthetic smoke tests. Opt in with RUN_ML_TESTS=1; no weights download."""
import gc
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from thyroid_poc.data import create_split, load_rgb


@unittest.skipUnless(os.environ.get("RUN_ML_TESTS") == "1", "Set RUN_ML_TESTS=1 for TensorFlow smoke tests")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        cls.tf = tf

    def tearDown(self):
        self.tf.keras.backend.clear_session()
        gc.collect()

    def test_train_reload_evaluate_and_ensemble(self):
        from thyroid_poc.train import train_model
        from thyroid_poc.evaluate import evaluate_models, write_report
        from thyroid_poc.artifacts import save_metadata, load_metadata
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source"
            for label, name in enumerate(("synthetic_a", "synthetic_b")):
                (source / name).mkdir(parents=True)
                for i in range(6):
                    Image.new("RGB", (16, 16), (label*180, i*20, 40)).save(source / name / f"{i}.png")
            manifest = create_split(source, root / "split", independent_images=True)
            model_path = root / "trained.keras"
            model = train_model(root / "split", model_path, epochs=2, batch_size=4)
            self.assertEqual(2, load_metadata(model_path)["training"]["epochs_completed"])
            sample = load_rgb(source / "synthetic_a/0.png", (224, 224))[None]
            before = np.asarray(model(sample, training=False))
            loaded = self.tf.keras.models.load_model(model_path, compile=False)
            np.testing.assert_allclose(before, np.asarray(loaded(sample, training=False)), rtol=1e-5, atol=1e-6)
            report = evaluate_models([model_path], root / "split", batch_size=2)
            expected = sum(r["split"] == "test" for r in manifest["rows"])
            self.assertEqual(expected, len(report["samples"]))
            second = root / "second.keras"
            loaded.save(second)
            save_metadata(second, manifest, size=(224, 224), architecture="roundtrip_fixture", seed=42)
            ensemble = evaluate_models([model_path, second], root / "split", batch_size=2)
            self.assertIn("ensemble", ensemble["metrics"])
            for row in ensemble["samples"]:
                p = row["probabilities"]
                self.assertAlmostEqual(p["ensemble"], (p["model_1"] + p["model_2"])/2)
            write_report(ensemble, root / "report.json")
            with self.assertRaises(FileExistsError): write_report(ensemble, root / "report.json")
            # Editing a dataset's labels/split cannot silently create test history.
            meta = load_metadata(second)
            meta["trained_split_hashes"].append(ensemble["samples"][0]["sha256"])
            second.with_suffix(".metadata.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError): evaluate_models([second], root / "split")

    def test_each_builder_forward_without_pretrained_download(self):
        from thyroid_poc.models import build_model
        for name in ("alexnet", "inception_v3", "inception_resnet_v2"):
            size = 224 if name == "alexnet" else 75
            model = build_model(name, (size, size, 3), pretrained=False)
            result = np.asarray(model(np.zeros((1, size, size, 3), dtype=np.float32), training=False))
            self.assertEqual((1, 1), result.shape)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(model.trainable)
            for layer in model.layers:
                if isinstance(layer, self.tf.keras.Model): self.assertTrue(layer.trainable)
            del model
            self.tf.keras.backend.clear_session()
            gc.collect()


if __name__ == "__main__":
    unittest.main()
