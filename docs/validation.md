# Software validation

Validated on 2 October 2026, Windows x64 CPU, Python 3.12.14. The tested environment used TensorFlow 2.20.0, Keras 3.15.1, NumPy 2.2.6 and Pillow 12.0.0. `requirements-runtime.txt` records the installed dependency versions. No GPU, ultrasound data or pretrained downloads were used.

| Check | Observed result |
|---|---|
| Core contracts | 18 tests passed; TensorFlow/OpenCV imports explicitly blocked during a separate core-only run |
| Full suite | 20 tests passed, including both runtime tests |
| Architecture smoke | AlexNet-style at 224×224; InceptionV3 and InceptionResNetV2 at their supported 75×75 minimum; random weights, finite binary outputs |
| Training integration | 12 generated RGB images, two classes; two full AlexNet-style epochs; metadata confirms two completed epochs |
| Serialization | Predictions before/after `.keras` save/reload agree within rtol 1e-5 / atol 1e-6 |
| Evaluation integration | Held-out manifest rows retained in order; individual and mean-ensemble reports saved; overlap and report overwrite rejected |

The group/duplicate tests cover reproducible output, unchanged sources, immutable destinations, mixed-class groups, connected exact copies, conflicting duplicate labels, too few groups, absent/extra group mappings, tampered/missing/untracked samples and unsafe relative paths. Metrics tests check a known confusion matrix/AUC, tied scores, single-class undefined metrics, invalid values, ensemble alignment and 50 deterministic pairwise-AUC oracle comparisons.

CI is configured to repeat core tests on Python 3.11/3.12 and synthetic runtime tests on Linux/Windows with Python 3.12. This local record is not a claim that remote CI has run; its status is shown by the actual workflow runs.

TensorFlow/Keras emit upstream deprecation messages during save/reset and end-of-sequence informational logs during dataset iteration. The final run completed both requested epochs and all tests. No warnings were converted into success claims about model quality.

## Not established by these checks

- Real ultrasound discrimination, patient-independent accuracy, calibration, fairness or clinical usefulness.
- Reproduction of the historical paper's metrics or exact architectures.
- A verified licence, version, labels or patient mapping for a real dataset.
- ImageNet download/fine-tuning quality, GPU behaviour or bit-identical results across platforms.
- A globally optimal grouped split; allocation is deterministic and approximate.

The retained historical figures are described separately in the README. Synthetic results are deliberately not presented as medical accuracy or benchmark scores.
