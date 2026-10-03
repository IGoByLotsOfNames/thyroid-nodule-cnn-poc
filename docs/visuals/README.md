# Reading the figures

## Recorded five-seed measurement

[measured-results.png](measured-results.png) and its [vector version](measured-results.svg) are generated from [the public aggregate](../../examples/recorded-measurement.json).

- The left panel includes every planned training seed on the same held-out split. Its dashed reference is the fixed-0.5 threshold result; the pale line marks the five-seed mean. The sample standard deviation describes variation between those fits.
- The right panel compares validation-selected and fixed thresholds on paired predictions. Its approximate 95% whole-group bootstrap interval conditions on the fitted models and thresholds; it does not include retraining uncertainty. The interval includes zero.
- This is the source-TIRADS experiment documented in the [model card](../../MODEL_CARD.md). It is not a diagnostic score, a five-model ensemble, or a comparison with the historical study.

The layout and study descriptors are specific to the October 2026 experiment. To regenerate the figures, use Python 3.12 in a separate plotting environment and run from the repository root:

```bash
python -m pip install matplotlib==3.10.6
python docs/visuals/plot_measurement.py
```

Matplotlib is only needed to regenerate the figures; the default project demo has no external dependencies. The script reads aggregate numbers only and does not load medical images or fitted models.

## Earlier framework diagrams

[experiment-flow.png](experiment-flow.png) describes the general library, including its optional architectures and ensemble evaluator. [experiment-records.png](experiment-records.png) explains artifact relationships and class-order semantics. They are preserved engineering diagrams, not depictions of the five separate fits being combined into an ensemble. The [current README diagram](../../README.md#from-data-to-evaluation) shows the completed experiment's specific ordering.
