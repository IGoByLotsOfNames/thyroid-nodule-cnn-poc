# Data statement

No medical images, annotations, patient identifiers or hospital submissions are included in this repository.

To reproduce the public proof of concept:

1. Identify the exact public dataset used in the original study.
2. Confirm its current licence and permitted uses.
3. Download it directly from the authorized source.
4. Record the dataset version, retrieval date, class definitions and exclusions.
5. Place it outside Git and create deterministic splits with a recorded seed.

Never commit raw medical data, `.keras`/`.h5` models, pickles or private storage credentials. The `.gitignore` file blocks common forms of these files, but staged changes must still be reviewed manually.

