# Model card: research software, no released clinical model

## What is available

Source code for binary AlexNet-style, InceptionV3 and InceptionResNetV2 models, group-aware dataset preparation, model metadata and per-sample individual/ensemble evaluation. No trained weights are distributed. Class order is dataset-specific and recorded explicitly; output is the probability of index 1.

## Intended use

Educational review of CNN experiments and reproducible ML software. A legitimate study requires verified data rights, label semantics, grouping and an appropriate evaluation protocol. The maintained code was tested on generated images only.

## Historical results

Archived cropped/uncropped accuracies of 0.88/0.93 used 502/482 evaluation images with fine-tuning overlap, as stated on pages 4 and 6 of the journal. They are in-sample historical summaries, not held-out estimates. Some historical AUC values were computed from hard predictions. Do not attach these results to the maintained implementation or treat them as clinical evidence.

The historical report describes four models; this repository reconstructs three architectures with different code and preprocessing. Its random-weight, synthetic smoke tests measure software correctness only. There is no newly established medical performance result.

## Evaluation contract

Models accept RGB float32 pixels in [0,255]; model layers own normalization. Training uses only train/validation partitions for fitting and early stopping. Evaluation requires the recorded manifest and model sidecar, rejects known train/validation overlap, and saves sample-level probabilities, hashes and ordered labels. Probability AUC, class precision/recall/F1, sensitivity/specificity, Brier score and log loss accompany a labelled confusion matrix. Undefined quantities are null. Ensemble probabilities use an unweighted mean. Model/threshold selection must happen before looking at the test results.

## Out-of-scope use and limitations

No diagnosis, screening, triage or treatment recommendations. No hospital deployment, prospective study, external validation, patient-level independence, calibration across sites, fairness or demographic/device subgroup result is demonstrated. Exact public-data provenance is unresolved; see [DATA.md](DATA.md). Manifest checks cannot repair incorrect group IDs or label definitions. Clinical use would require separate governance, expert oversight and independent validation.
