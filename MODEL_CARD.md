# Model card

## Model details

Binary image-classification proof of concept comparing custom AlexNet-style and transfer-learning CNNs, with experiments combining model probabilities into an ensemble.

## Intended use

- Educational reproduction of the research pipeline
- Exploration of preprocessing, CNN training and ensemble evaluation
- Documentation of applied machine-learning research

## Out-of-scope use

- Diagnosis, screening, triage or treatment recommendations
- Use on hospital data without governance and validation
- Claims of performance beyond the documented experiment

## Training and evaluation data

The retained experiment used public/Kaggle ultrasound images organized into benign and malignant classes. The data is not redistributed here. Exact dataset provenance and licence must be confirmed before public release.

## Metrics

The retained reports include precision, recall, F1, confusion matrices, accuracy and ROC AUC. One cropped-image experiment reported 0.88 accuracy and 0.882 AUC; one uncropped-image experiment reported 0.93 accuracy and 0.925 AUC.

## Limitations and risks

The archive does not contain the evidence required to claim clinical generalization, calibration across hospitals or equitable performance across demographic and equipment subgroups. Ultrasound-image classification can be affected by acquisition conditions, labelling practice, prevalence and dataset leakage. Any future clinical study requires independent validation, governance and qualified medical oversight.
