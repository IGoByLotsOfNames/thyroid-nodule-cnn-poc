# Data provenance and split protocol

The repository distributes source, synthetic demo inputs and aggregate experiment results. It does **not** distribute medical images, XML annotations, group manifests, sample-level predictions, hospital submissions or trained medical weights.

## Original study and pinned reconstruction

The original journal cites [DASMEHDIXTR's DDTI Thyroid Ultrasound Images on Kaggle](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images). The later engineering experiment pins **version 1, dated 29 July 2021**, using this archive SHA-256:

```text
c6bc36f4b555008073333f0aa4888ddc5f51fc90e23ee1a8a0512e770a5ea7d3
```

The inspected snapshot contains 390 XML cases and 480 JPG images. This reconstructs a source named by the study; it does not prove that these are the exact download bytes used in the original experiments.

Source metadata labels the licence **“Other”**. A standardized redistribution grant for these medical assets has not been established, so raw and derived images, medical weights and sample-level records remain local. The repository's own [licence](LICENSE) does not grant rights over third-party data. Anyone repeating the experiment must verify their own permitted access and use.

## Label contract and exclusions

The current target uses the source's TIRADS annotation:

| Class | Included source categories |
|---|---|
| `0_tirads_2_3` | 2, 3 |
| `1_tirads_4a_4b_4c_5` | 4a, 4b, 4c, 5 |

These are source risk categories, not pathology-confirmed cancer labels or an assertion of modern ACR TI-RADS equivalence. Missing annotations are excluded. Unsupported nonempty labels are rejected rather than silently reassigned.

The resulting cohort retains **349 images from 298 labelled cases**, with 61 class-0 and 288 class-1 images. **131 images from 92 cases** are excluded for missing annotations. Full case and view identities are preserved rather than inferred from a filename's first digit. The original study's labels and preprocessing must not be assumed equivalent to this explicit target.

## Group before splitting

The importer connects source case identities and exact byte/decoded-RGB duplicates across the complete source graph **before excluding missing-label cases**. An excluded case can therefore preserve a connection between two included cases. Mixed-label groups stay together; conflicting labels for identical bytes or decoded pixels reject.

The splitter allocates whole connected components to train, validation and test. Split seed `20261002`, with requested proportions 70/15/15, yielded:

| Partition | Images | Class 0 | Class 1 | Declared groups |
|---|---:|---:|---:|---:|
| Train | 245 | 43 | 202 | 209 |
| Validation | 52 | 9 | 43 | 45 |
| Test | 52 | 9 | 43 | 44 |

Allocation is approximately stratified; indivisible groups make the requested ratios approximate. Every partition must contain both classes. Missing group mappings, conflicting duplicate labels, unusable class coverage and existing destinations are rejected. Source images are never moved or modified.

The recorded split has no detected cross-partition case, connected-group, exact-byte or decoded-RGB overlap. These are **declared groups, not verified unique patients**. Near duplicates, crops, augmentations or repeated patients not represented by those links may remain. Correct grouping and external validation require additional domain knowledge; hashes cannot provide it.

## Traceable inputs and evaluation

Manifests record class order, seed, requested proportions, grouping policy, source fingerprint and each image's relative path, group, label, size and SHA-256. Models retain their training/validation bindings and a metadata sidecar. Training and evaluation recheck the recorded inputs and reject additions, removals, changes or known split leakage.

The experiment plan freezes the split, recipe, training seeds, code and runtime. Every fit and validation threshold selection finishes before test prediction; an evaluation lock seals the resulting model and selection artifacts. Integrity validation can inspect held-out bytes and manifest labels. It does not imply the test set was inaccessible or that hashes prove researcher behaviour.

For the completed experiment, the metric unit is an **image**. The bootstrap resamples whole declared groups to preserve their observed dependence, but this does not turn the results into patient-level metrics or establish independent patients. The [model card](MODEL_CARD.md) and [measurement protocol](docs/measurement.md) describe this distinction and the small class-0 support.

## Using another dataset

Before preparing a new study, document the authorized source and version, label definition, exclusions, source-image/patient grouping, permitted uses and acquisition limitations. Keep repeated views, crops and derived images together according to that protocol; apply augmentation only after splitting. A generic class-folder name does not establish a diagnosis.

Create a fresh experiment directory with choices fixed before test evaluation. Keep completed records intact, retain every planned seed and investigate failed integrity checks instead of editing checksums to make them pass. The existing evaluator binds results to a recorded dataset; external validation needs its own protocol.

Do not publish dataset roots, medical manifests, group CSVs, sample-level reports, medical model files, tokens or credentials. `.gitignore` helps avoid accidents but does not determine whether data may be shared. Inspect the actual publication contents.
