# Fixed-split threshold measurement

**User-run measurement completed on 3 October 2026.** The saved five-seed result reports mean test balanced accuracy 55.79% (sample SD 5.64 percentage points), versus 50.00% at the fixed threshold. The mean paired difference is +5.79 percentage points; its approximate conditional group-bootstrap interval is [-0.25, +12.15] percentage points and includes zero. This does not establish a reliably positive improvement. See the [aggregate copy](../examples/recorded-measurement.json) and the [validation record](validation.md). The protocol below describes the frozen experiment; do not rerun it to choose a more favorable result.

This experiment asks whether selecting a decision threshold on validation images changes held-out, image-level balanced accuracy compared with a fixed threshold of 0.5. Each comparison uses the **same fitted model and the same test probabilities**. The architecture, data split and training recipe stay fixed; five prespecified training seeds show how much the result varies with training randomness.

## Frozen target and training recipe

The local public-DDTI reconstruction uses source TIRADS **2/3 as class 0** and **4a/4b/4c/5 as class 1**. Missing annotations are excluded; unsupported nonempty annotations reject. These are source risk categories, not pathology-confirmed benign/malignant diagnoses or an assertion of modern ACR TI-RADS equivalence. Full images are retained. Case identities and exact byte/decoded-RGB duplicate links were connected before splitting, including links through excluded cases.

The frozen local split records the following counts. Its aggregate support is included with the [recorded measurement](../examples/recorded-measurement.json).

| Partition | Images | Class 0 | Class 1 | Declared groups |
|---|---:|---:|---:|---:|
| Train | 245 | 43 | 202 | 209 |
| Validation | 52 | 9 | 43 | 45 |
| Test | 52 | 9 | 43 | 44 |

These are the recorded counts for split seed `20261002`, not model results. Treating declared groups as independent is not verified patient independence; undetected near-duplicates or repeated patients may remain. The dataset is local and is not distributed with the repository. See [data provenance](../DATA.md).

| Setting | Prespecified value |
|---|---|
| Training seeds | `17, 29, 43, 71, 101`; retain every seed |
| Model | Current AlexNet-style CNN with batch normalization and global average pooling; [source](../src/thyroid_poc/models.py) |
| Images | Full-image Pillow RGB conversion and bilinear resize to 224 × 224; model rescales values by 1/255 |
| Optimization | Adam, learning rate `1e-4`, binary cross-entropy, batch size 16 |
| Stopping | At most 30 epochs; early stopping on `val_loss`, patience 5, restore best weights |
| Additional choices | No pretrained weights, augmentation, class weighting or ensemble |
| Runtime | Separate CPU worker per seed/phase; two intra-op and one inter-op TensorFlow threads; deterministic operations requested |

The frozen plan records source hashes, the exact dataset manifest, runtime package versions and these choices. A changed plan, code, runtime or dataset fails validation rather than silently continuing a different experiment. Seeds and deterministic settings aid replay; they do not promise identical numbers across all hardware or libraries.

The runner completes **all five fits and all five validation threshold selections before any test prediction**. Early stopping already uses validation loss, so the later validation balanced accuracy is a selection score, not an unbiased performance estimate. Threshold candidates are the distinct validation probabilities plus 0, 0.5 and 1; predictions are positive when `p >= threshold`. Ties prefer the candidate closest to 0.5, then the smaller candidate. The final artifacts are sealed in an evaluation lock before test workers start. Integrity validation may read held-out file bytes and manifest labels; the barrier prevents test inference and test-driven choices during fitting and selection.

Fit checkpoint validation requires each model sidecar's seed to be an integer equal to its planned seed. The completed run used the frozen code and runtime recorded in its plan. Artifact hashes establish recorded consistency, not proof that a person never inspected or edited evidence.

## What the completed report measures

Balanced accuracy is the mean of class-0 recall and class-1 recall. It prevents the larger class from dominating the primary score. The [official metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) matches this implementation.

For each training seed, the report recomputes test balanced accuracy from aligned probabilities at the frozen validation-selected threshold and at 0.5. The primary paired effect is:

`delta(seed) = balanced_accuracy(selected threshold) - balanced_accuracy(0.5)`

It reports all five paired values, both absolute scores, their means, and sample standard deviation and variance with denominator `5 - 1` (`ddof=1`). A positive observed mean is a descriptive improvement on this fixed test split; it is not by itself evidence of broad superiority. Do not choose the winning seed or revise architectures, epochs or thresholds after seeing test results.

A separate constant baseline learns its positive probability from **training images only**: positive training count divided by training count, then a fixed 0.5 threshold. It ignores image features. Its balanced accuracy is 0.5 whenever both test classes are present; its apparent ordinary accuracy can still be high on an imbalanced set. This follows the distinction between a majority prediction and empirical prior probabilities in [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html). No extra classifier dependency is needed.

Seed SD and variance describe observed algorithmic variability **conditional on this split and recipe**. Five seeds are not five fresh test sets, and their images must not be pooled into 260 independent observations. The seed mean is also not the accuracy of an ensemble. Dataset sampling, model selection and training randomness are different sources of uncertainty, as discussed by [Bouthillier et al.](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf).

The report also computes an approximate, paired group-bootstrap interval:

- Use Python's seeded RNG with seed `1729`. Draw 44 declared test groups with replacement per replicate, retaining every image and each repeated group's full multiplicity. The metric remains image-level; groups are the resampling unit.
- Reuse each draw for every fitted seed, both threshold choices and the constant baseline. Average the per-seed metrics or paired differences on that draw; do not average model probabilities into an ensemble.
- Request 2,000 valid draws. A draw missing either true class has undefined balanced accuracy and is rejected. Record attempts and rejections; stop after 40,000 attempts (`20 × 2,000`). If the valid target is not reached, record `insufficient_valid_draws` with null intervals and a nonzero CLI status.
- Use the 2.5th and 97.5th percentiles with linear interpolation. These are approximate 95% percentile intervals conditional on these fitted models, thresholds and resamples retaining both classes. They do not include retraining uncertainty.

The method treats declared groups as approximately independent. Cluster-bootstrap validity depends on the data's dependence structure; it does not establish patient independence. [Field and Welsh](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2007.00593.x) explain why the model of clustering matters. Unresolved exact-byte links across declared test groups reject rather than treating those groups as independent.

Only nine class-0 images are available in each of validation and test. On the test set, changing one class-0 prediction moves class-0 recall by 11.11 percentage points and balanced accuracy by 5.56 points. Small class counts make thresholds and intervals unstable. No result here establishes calibration, clinical utility, pathology prediction, external-site performance or unique-patient generalization. Historical cropped/uncropped study figures are not a before/after baseline for this different target and protocol.

## Inspecting the recorded measurement

The public [aggregate JSON](../examples/recorded-measurement.json) retains every seed, both threshold policies, support counts, summary statistics, bootstrap results and provenance hashes. The offline [demo](demo.md) renders that aggregate without training or inference. Medical images, fitted models, per-image reports and local machine paths are not redistributed.

The completed private experiment contains a frozen `plan.json`, an evaluation lock, verified fit/test checkpoints, model metadata, threshold selections and aligned probabilities. Its retained records were checked by recomputing validation decisions and metrics and verifying artifact hashes; the check did not fit or infer again. The public aggregate makes the reported values inspectable, but cannot independently reproduce private inputs that are not distributed.

## Preparing a separate authorized experiment

Install the package and recorded Python 3.12 runtime as described in [validation](validation.md). After importing an authorized dataset and creating its validated split according to [DATA.md](../DATA.md), run:

```bash
python -m thyroid_poc.experiment prepare authorized-split experiment-run --seeds 17 29 43 71 101 --epochs 30 --batch-size 16 --bootstrap-repetitions 2000
python -m thyroid_poc.experiment run experiment-run
python -m thyroid_poc.experiment summarize experiment-run
```

`authorized-split` and `experiment-run` are local paths you supply, not bundled data. Preparation requires a fresh output directory and records choices before fitting. The final command recomputes the summary from verified saved evidence without new fitting or prediction. These commands describe a separate experiment; they do not reproduce the recorded numbers from the public repository alone.

Use one invocation at a time for an experiment directory. Wait for a previous process to exit before retrying. Keep its dataset, source and recorded runtime unchanged. The runner starts workers with the fixed CPU settings; do not call the internal `worker` command directly.

The console and each attempt's `process.log` show progress. A worker bundle is accepted only after the worker exits successfully and its artifacts are verified. Failed or interrupted attempts remain available. Re-running `run` reuses verified checkpoints and retries incomplete work for the same planned seed in a fresh attempt directory. Once the evaluation lock exists, changed or missing fits cannot be retrained inside that experiment. Investigate an integrity mismatch instead of editing checksums or deleting evidence to bypass it.

Retain every planned seed and failed attempt. Choose any new protocol before examining its test results; do not select a favorable rerun. The original historical cropped/uncropped figures concern a different target and protocol and are not a before/after baseline for this measurement.
