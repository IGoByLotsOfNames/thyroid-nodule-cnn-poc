# Software validation

The tests exercise the public pipeline on generated images and constructed records. They do not fit the medical dataset or establish clinical performance. The completed source-TIRADS measurement has a separate [protocol and aggregate record](measurement.md).

## Observed local checks

On **3 October 2026**, the integrated checkout was checked on Windows x64 CPU with Python 3.12.14. The full runtime used TensorFlow 2.20.0, Keras 3.15.1, NumPy 2.2.6, Pillow 12.0.0, Ruff 0.16.10 and coverage.py 7.16.2. The separate core environment used NumPy 2.2.6 and Pillow 12.3.0, with TensorFlow absent.

| Check | Observed result |
|---|---:|
| Full synthetic suite | 109 passed, 1 skipped (110 collected) |
| Core suite without TensorFlow | 104 passed, 6 skipped (110 collected) |
| Ruff lint and formatting | Passed; 27 Python files |
| Dependency consistency | Passed in both environments |
| Statement coverage | 1,350 / 1,447 = 93.30% |
| Branch coverage | 495 / 574 = 86.24% |
| Combined statement + branch coverage | 1,845 / 2,021 = 91.29% |
| Coverage exclusions | Zero excluded source lines |

All checks above were rerun on the integrated checkout. The fresh [machine-readable verification receipt](evidence/software-validation.json) records these counts and the exact source hashes. Subprocess coverage was combined with the parent process; the 90% floor applies to statements and branches together, not branch coverage alone.

The full-suite skip is native Windows symlink creation, which this process cannot perform. A simulated guard test passed. Core mode additionally skips five explicitly opt-in TensorFlow integrations. Neither skip is represented as a passing runtime check.

## Behaviours exercised

The core tests cover data/cohort identity, exact and decoded-pixel duplicate groups, incomplete mappings, changed inputs, class order, metadata bindings, malformed artifacts, output collisions, threshold selection against an exhaustive oracle, paired bootstrap arithmetic, checkpoint/resume barriers and report identity.

Five TensorFlow integrations execute:

1. Random-weight forward passes for all three model builders.
2. Synthetic training, save/reload agreement, individual evaluation and aligned ensembles.
3. Validation-only threshold selection followed by test evaluation, including model reordering.
4. A two-seed CPU experiment-worker lifecycle, phase ordering and idempotent resume.
5. The optional one-epoch synthetic CNN demo command.

The default demo is also run from another directory with isolated/no-site flags. It checks all sixteen generated PNGs and the handcrafted probability fixture, then writes self-contained HTML/JSON without TensorFlow. Its threshold and scores are deliberately constructed examples. Tests check exact input inventory, corrupt fixtures, validation-only selection, no-overwrite behaviour, text escaping and unsafe image-URI omission.

All twelve package source files retain the hashes used by the completed experiment plan. The public demo is outside that package and does not change its medical experiment. The journal and archived reports retain their original bytes.

## Medical measurement verification is separate

The user completed the frozen five-seed source-TIRADS experiment on 3 October 2026. Read-only verification recomputed validation decisions, individual test metrics and the aggregate and checked plan, source, split, model, metadata and checkpoint bindings. This verification did not retrain or perform new inference. Only the curated aggregate is public; medical images, fitted weights and per-image records remain local.

The metric is image-level balanced accuracy for source TIRADS categories. There are 52 test images in 44 declared groups, with class support 9/43. The approximate paired interval includes zero. These records do not establish pathology prediction, verified patient independence or external clinical validation. See [MODEL_CARD.md](../MODEL_CARD.md) and [measurement.md](measurement.md) for the complete context.

## Repeat the software checks

For core checks, use a fresh Python 3.11-3.13 environment without TensorFlow:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip check
python -c "import importlib.util; assert importlib.util.find_spec('tensorflow') is None"
python -m ruff check src tests demo.py demo_support
python -m ruff format --check src tests demo.py demo_support
python -B -m unittest discover -s tests -v
```

Use a separate Python 3.12 environment for the full CPU suite. These PowerShell commands enable branch and subprocess coverage, matching the workflow's scope:

```powershell
python -m pip install -r requirements-runtime.txt
python -m pip install -e ".[dev]"
python -m pip check
$env:RUN_ML_TESTS = "1"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:COVERAGE_RCFILE = (Join-Path (Get-Location) "pyproject.toml")
$env:COVERAGE_FILE = (Join-Path (Get-Location) ".coverage")
python -m coverage erase
python -m coverage run -m unittest discover -s tests -v
python -m coverage combine
python -m coverage xml --fail-under=0 -o coverage.xml
python -m coverage json --fail-under=0 -o coverage.json
python -m coverage report
```

On macOS/Linux, set the same variables with `export NAME=value`. Coverage must be installed into the active environment so its normal subprocess startup hook is available. JSON/XML export preserves diagnostics regardless of the floor; the final `coverage report` enforces it. The configured source includes all twelve package modules, `demo.py` and both `demo_support` modules. No coverage exclusions are configured.

## Continuous integration

[The workflow](../.github/workflows/python.yml) defines six jobs: one quality job; core tests on Linux with Python 3.11, 3.12 and 3.13; and full synthetic/coverage jobs on Linux and Windows with Python 3.12. Repository permissions are read-only, stale runs are cancelled and coverage reports are retained per platform. It does not run the private real-data experiment or download pretrained weights.

The prepared workflow passed static validation with actionlint 1.7.12. Hosted results are separate evidence: inspect [GitHub Actions](https://github.com/IGoByLotsOfNames/thyroid-nodule-cnn-poc/actions/workflows/python.yml) for the exact current commit. The local Windows results above do not imply success on another operating system or Python version.

## Remaining limits

- HTML structure, escaping and offline assets are tested automatically. Those checks do not substitute for browser appearance, responsive layout or print review.
- Native Windows symlink/junction behaviour, GPU execution and pretrained downloads are not covered by this local CPU run.
- The demo CLI and experiment-worker lifecycle execute successfully. Some standalone module-wrapper success paths and fault/concurrency scenarios remain outside the tested paths.
- An experiment directory supports one invocation at a time. Hashes reveal recorded inconsistencies; they do not establish trustworthy labels or prove that a person never inspected or altered evidence.
- Statistical and clinical generalization require separate evidence. The small minority class, unverified patient identity, possible near-duplicates and lack of external validation remain material limitations.

Third-party TensorFlow/Keras deprecation notices occurred during passing checks. Historical cropped/uncropped figures are in-sample research records, not results of this maintained implementation.
