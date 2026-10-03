# Inspectable evaluation demo

The default demonstration runs offline with Python's standard library. It checks bundled inputs, applies the checkout's threshold-selection and metric functions, and writes a self-contained HTML report plus JSON. It needs no package installation, TensorFlow, credentials, dataset download or fitted weights.

## One command

From the repository root, using Python 3.11 or later:

```bash
python demo.py
```

Open the printed `report.html` path. Every run creates a fresh `demo-output/run-*/` directory. Styles and images are embedded, so the report works without a server. `report.json` contains the same results in machine-readable form. You can choose a new destination with `--output PATH`; an existing destination is never overwritten.

The script locates inputs relative to itself, so it also runs from another working directory. For a strict no-site-packages check:

```bash
python -I -S -B demo.py
```

## Two separate evidence panels

**Synthetic walkthrough.** Sixteen generated colour tiles are bundled under `examples/images`: eight training placeholders, four validation examples and four test examples. Identities, partitions, labels and checksums are in [demo-input.json](../examples/demo-input.json). The probabilities are hand-authored to demonstrate threshold selection, not CNN predictions. The default mode verifies image bytes and recomputes the metrics without fitting or inference.

The constructed fixture selects threshold 0.700 using validation rows. Its synthetic test balanced accuracy is 75.00%, versus 50.00% at threshold 0.5; the selected confusion matrix is `[[1,1],[0,2]]` (rows true, columns predicted). These are deliberately constructed software examples, not ultrasound results.

**Recorded research.** [recorded-measurement.json](../examples/recorded-measurement.json) contains aggregate results and provenance hashes from the completed five-seed source-TIRADS experiment. It contains no medical images, patient/case IDs, local account paths, weights or per-image predictions. The report shows all five seeds and the approximate paired interval, including its crossing of zero. This panel displays saved results; it performs no new experiment or ensemble prediction. Read the [measurement protocol](measurement.md) for definitions and limitations.

## Optional CNN execution on generated images

With the Python 3.12 full runtime installed as in [validation](validation.md):

```bash
python demo.py --cnn
```

This adds one CPU training epoch of the AlexNet-style builder on the sixteen synthetic PNGs, with seed 17 and batch size 4. It creates a split, fits and saves a model, reloads it, selects a validation threshold and predicts held-out generated images. The split is generated afresh and differs from the default handcrafted fixture split. The model, split manifest, selection and evaluation files stay inside that run's `cnn-smoke/` directory.

There are no downloads or pretrained weights. The smoke score is not expected to be useful and has no medical interpretation; successful execution of the pipeline is the check. If dependencies are missing, the command stops before creating output and explains what is needed. It never installs packages automatically.

## Implementation and checks

- [demo.py](../demo.py) owns input verification, validation-only threshold selection and fresh output creation.
- [demo_support/report.py](../demo_support/report.py) escapes text, restricts embedded image data and renders static HTML.
- [test_demo.py](../tests/test_demo.py) checks corrupt inputs, changed inventory, partition boundaries, output collisions, escaped text and the optional CPU path.

The demo is separate from the frozen package source, preserving the completed experiment's source hashes. Checksums establish internal consistency, not authenticity, valid labels or patient independence. The recorded research has 52 test images in 44 declared groups, including only nine class-0 images; its intervals are approximate and conditional on the fitted models.

Screenshots should keep the synthetic and recorded-research labels visible. Automated HTML checks and numerical tests are separate from visual/browser verification; see the [validation record](validation.md) for observed checks.
