# Model card: thyroid-ultrasound research pipeline

## Scope

This repository contains research software and aggregate experiment records, not a released clinical model. It supports binary AlexNet-style, InceptionV3 and InceptionResNetV2 models, grouped dataset preparation, model metadata and individual or probability-mean ensemble evaluation. No trained medical weights are distributed.

The completed experiment below uses **five separate AlexNet-style fits on one fixed split**. It does not evaluate an ensemble or compare the three available architectures. The AlexNet-style builder uses batch normalization and global average pooling; it is not an exact reproduction of the original AlexNet architecture.

The intended use is educational review of ML engineering and research methods. Diagnosis, screening, triage and treatment decisions are outside scope.

## Task and data

The 3 October 2026 experiment uses a pinned reconstruction of DASMEHDIXTR's DDTI Kaggle snapshot, with these source annotations:

| Index | Recorded class name | Source TIRADS categories |
|---|---|---|
| 0 | `0_tirads_2_3` | 2 and 3 |
| 1 | `1_tirads_4a_4b_4c_5` | 4a, 4b, 4c and 5 |

These are **source risk categories**, not pathology-confirmed benign/malignant diagnoses or a conversion to modern ACR TI-RADS. Missing annotations are excluded; unsupported nonempty annotations are rejected. Full ultrasound images are used. Case identities and exact byte/decoded-RGB duplicate links are connected before exclusions, including links through excluded cases.

The retained cohort contains 349 images from 298 labelled cases. Split seed `20261002` produced:

| Partition | Images | Class 0 | Class 1 | Declared groups |
|---|---:|---:|---:|---:|
| Train | 245 | 43 | 202 | 209 |
| Validation | 52 | 9 | 43 | 45 |
| Test | 52 | 9 | 43 | 44 |

There is no recorded cross-partition overlap in cases, connected groups, exact bytes or decoded-RGB hashes. These controls do not establish unique-patient independence: unidentified repeated patients, near duplicates and derived images may remain. Dataset provenance, exclusions and redistribution limits are in [DATA.md](DATA.md).

## Frozen training and selection

| Setting | Value |
|---|---|
| Training seeds | 17, 29, 43, 71, 101; every planned seed retained |
| Input | Full-image RGB, Pillow bilinear resize to 224 × 224; model rescales by 1/255 |
| Initialization | Random weights |
| Optimization | Adam, learning rate `1e-4`, binary cross-entropy, batch size 16 |
| Stopping | At most 30 epochs; early stopping on validation loss, patience 5, restore best weights |
| Additional choices | No pretrained weights, augmentation, class weighting or ensemble |
| Execution | Separate CPU worker per seed and phase; two intra-op and one inter-op TensorFlow threads; deterministic operations requested |

The plan binds the recipe, split, source and runtime before execution. All five fits and validation threshold selections complete before test prediction. Integrity checks may read held-out bytes and manifest labels; they do not make the test partition inaccessible.

Threshold candidates are the distinct validation probabilities plus 0, 0.5 and 1. A sample is positive when `p >= threshold`. Selection maximizes validation balanced accuracy, breaking ties by closeness to 0.5 and then the smaller threshold. Validation is also used for early stopping, so its selection score is not an unbiased performance estimate. An evaluation lock seals the final artifacts before test workers begin.

## Completed results

The experiment was run by the author on 3 October 2026. The saved artifacts and metrics were checked locally; creating the offline demonstration did not retrain the models. Balanced accuracy is the mean of the two class recalls. Each paired comparison uses the same fitted model, test images and probabilities.

| Seed | Selected threshold | Selected balanced accuracy | Fixed-0.5 balanced accuracy | Paired difference |
|---|---:|---:|---:|---:|
| 17 | 0.572651 | 51.68% | 50.00% | +1.68 pp |
| 29 | 0.572537 | 53.10% | 50.00% | +3.10 pp |
| 43 | 0.609592 | 51.81% | 50.00% | +1.81 pp |
| 71 | 0.674012 | 57.36% | 50.00% | +7.36 pp |
| 101 | 0.799536 | 64.99% | 50.00% | +14.99 pp |

Across the five fits, mean selected balanced accuracy is **55.79%**, sample SD **5.64 percentage points** (`ddof=1`) and variance **0.00317606** on the 0–1 scale. The approximate conditional 95% interval for the mean is **49.75–62.15%**. The mean paired difference from threshold 0.5 is **+5.79 percentage points**, with approximate conditional interval **[−0.25, +12.15] percentage points**. Because that interval includes zero, these results do not establish a reliably positive improvement.

At threshold 0.5, every fitted model predicts class 1. The confusion matrix is `[[0, 9], [0, 43]]`, with true classes as rows and predictions as columns. This gives ordinary accuracy 82.69% but balanced accuracy 50%. A separate constant baseline learns the positive prevalence from training images only (`202 / 245`) and also has balanced accuracy 50% at 0.5.

The seed SD describes variation in training on this fixed split. It is not a confidence interval, an ensemble score or evidence from five independent test cohorts. The 52 images must not be counted five times as 260 independent observations.

### Uncertainty and resolution

The paired bootstrap uses 2,000 shared whole-group resamples, seed `1729`, drawing 44 declared test groups with replacement and retaining each selected group's full image multiplicity. Every draw is reused across all fitted seeds, both thresholds and the baseline. Metrics remain image-level; groups are the resampling unit. In this run, all 2,000 draws retained both classes.

Intervals are the 2.5th and 97.5th percentiles with linear interpolation, conditional on the fitted models, thresholds and resamples retaining both classes. They do not include retraining uncertainty. Their interpretation assumes declared groups are approximately independent; that assumption is not verified patient independence.

There are only nine class-0 test images. Changing one class-0 prediction changes balanced accuracy by **5.56 percentage points**. Small class counts make threshold choices and intervals unstable. No external-site, prospective, pathology-confirmed, calibration, fairness or device/demographic subgroup evaluation is established here.

The [machine-readable aggregate](examples/recorded-measurement.json) retains all five fits, intervals, support counts and artifact hashes. The [measurement guide](docs/measurement.md) explains the protocol and replay commands.

## Evaluation and artifact contracts

Models accept RGB float32 pixels in `[0, 255]`; normalization belongs inside the serialized model. Sorted class names define indices, and sigmoid output always means probability of index 1. Training uses only train/validation for fitting and early stopping.

Evaluation validates the recorded manifest, model hash and metadata sidecar, rejects known training/validation overlap and saves aligned probabilities and ordered labels. Reports include a labelled confusion matrix, class precision/recall/F1, probability ROC AUC, sensitivity/specificity, Brier score and log loss. Undefined quantities are `null`; reports refuse silent overwrite. The library's optional ensemble is the unweighted mean of aligned probabilities.

The experiment runner accepts a checkpoint only after a successful worker exit and artifact validation. It retains failed attempts and resumes verified completed phases; it does not resume midway through an epoch. After evaluation is locked, changed or missing fits cannot be retrained within that experiment. Use one runner at a time for an experiment directory. Hashes establish consistency with recorded artifacts, not authenticity or proof that nobody inspected or edited the records.

## Historical study and later development

The original [journal](docs/proof-of-concept-journal.pdf) documents the author's work at KMUTT's ESIC PLUS Laboratory. It describes a four-model ensemble and acknowledges the professor who suggested whole-scan inputs, together with the professors and seniors who guided the study. The maintained three-builder implementation is a later reconstruction, not the exact historical implementation.

The archived cropped/uncropped reports show accuracies of 0.88/0.93 on 502/482 evaluation images. **Pages 4 and 6 state that evaluation reused fine-tuning images.** These are in-sample historical summaries, not held-out estimates. Cohorts differ, and some historical AUC values used hard predictions. They neither establish a controlled whole-scan advantage nor form a before/after comparison with the new source-TIRADS task and balanced-accuracy metric. The [historical result images](results/README.md) remain unchanged.

The author reports that later development in the wider research project introduced hospital data, progressed to deployment in Thai hospitals and included a private repository through which hospitals could contribute further training data. This rebuild has not independently verified those later developments, the private system's behaviour or its clinical outcomes. The personal contribution documented here is the original proof of concept and the later public engineering rebuild; no hospital data or private submission system is included.

## Demonstrations and software checks

The default demo generates colour images and uses handcrafted probabilities. Its constructed 75% balanced accuracy is a teaching example, not CNN performance or a medical result. Optional `--cnn` mode trains an AlexNet-style model for one epoch on generated images to exercise the code; it promises no medical accuracy. The recorded real-data aggregate is presented separately in both documentation and the default report.

The [validation record](docs/validation.md) reports the dated software environments, tests, coverage and skipped checks. Passing software tests supports implementation behaviour, not clinical suitability. Medical use would require separate data governance, expert oversight and independent clinical validation.
