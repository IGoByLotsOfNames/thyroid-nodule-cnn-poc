# Data provenance and split protocol

No medical images, annotations, patient identifiers, hospital submissions or trained models are distributed here. Tests generate artificial colour images locally.

The original journal names **DASMEHDIXTR's DDTI Thyroid Ultrasound Images on Kaggle**. This is a provenance lead, not verification of the exact downloaded version, upstream licence, label derivation or patient grouping. The historical archive also contains preprocessing revisions whose first-digit filename parsing and handling of uncertain TIRADS labels require reconstruction. Do not reuse those labels uncritically or infer that a risk category is a pathology-confirmed diagnosis.

Before a real-data experiment, record and verify:

1. Authorized upstream source, dataset version/checksum, licence and permitted uses.
2. Label definition, source and handling of unknown/ambiguous labels; exclusions with reasons.
3. Patient/case/source-image grouping, including crops, repeated views and augmented copies.
4. A frozen group CSV and split seed; which data is used for training, model selection and final evaluation.
5. Study population, acquisition limitations and any required institutional permission.

The implemented splitter sorts files, hashes bytes, connects explicit groups with duplicate hashes, and allocates whole components to partitions. Conflicting labels for identical bytes are rejected. A seeded greedy allocation targets per-class ratios while prioritizing coverage; this is **approximate stratification**, not an optimizer or a guarantee of feasibility for every grouped dataset. Unusable class coverage is rejected. It requires a new destination and stages a complete validated output before renaming it. It never moves or modifies source images.

`manifest.json` records class names, seed, requested proportions, grouping policy, source fingerprint and each image's relative source path, destination, group, label, byte count and SHA-256. Training and evaluation recheck every sample and reject additions, removals, tampering or group/hash leakage. Content hashes detect exact byte copies only. Different encodings and derived images must share an explicit group. Unknown grouping cannot support a patient-independent generalization claim.

Models record hashes of both training and validation images; evaluation rejects overlap with held-out images. A supplied manifest or metadata file is an experiment record, not a digital signature or proof that a researcher used correct grouping. Keep manifests and reports private when filenames, opaque IDs or derived values are sensitive.

Do not commit dataset roots, medical manifests, group CSVs, model files, pickles, tokens or credentials. `.gitignore` is a convenience, not a data-governance control. Check staged files before publication.
