# Original research results

These unchanged figures belong to the original four-model proof-of-concept study at KMUTT's ESIC PLUS Laboratory:

- [Cropped-image ensemble report](cropped-ensemble-classification-report.png): reported accuracy 0.88 on 502 evaluation images.
- [Uncropped-image ensemble report](uncropped-ensemble-classification-report.png): reported accuracy 0.93 on 482 evaluation images.

Pages 4 and 6 of the [original journal](../docs/proof-of-concept-journal.pdf) state that evaluation reused fine-tuning images. These are **in-sample historical summaries, not held-out accuracy**. The cohorts differ, and some archived AUC calculations used hard predictions. They do not establish a controlled advantage for whole-scan inputs.

The current source-TIRADS experiment uses a different target, implementation, split and metric. Its [model card](../MODEL_CARD.md#completed-results) records all five fits and their uncertainty. Comparing its balanced accuracy directly with these historical accuracy figures would not measure improvement or deterioration.
