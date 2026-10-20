# AquaTriage application reference v1

This reference supports prediction and review of new chemical-species-test
cases. It contains the fitted preprocessing, five 400-tree LightGBM models,
and the training-case representations used for support distances.

| Property | Value |
| --- | --- |
| Training records / chemicals | 4519 / 775 |
| Target | log10(mol/L), acute fish, daphnid and algal LC50/EC50 |
| Fingerprint | Morgan, radius 2, 2048 bits |
| Physicochemical descriptors | Seven RDKit descriptors, including MolLogP |
| Bioactivity predictors | 34 variables |
| Publication year | Not a predictor or reliability input |
| Ensemble members / trees per member | 5 / 400 |
| Seed | 42 |

The training set is the identity-corrected primary benchmark. Data came from
ECOTOX, with curated chemical identity and bioactivity linkage; the benchmark
and its provenance are distributed in the project's analysis release.
This application reference was fitted on the full benchmark, not on a held-out
validation split. Its use does not constitute a new performance evaluation.

`reference.json` records the model configuration, training-data fingerprint,
software versions, and SHA-256 checksum of `reference.joblib`. The hosted app
verifies the checksum before loading the model. The model contains processed
training representations and case identifiers for nearest-neighbor reporting.

Use Python 3.11 and the versions in `app/requirements.txt`. Only load joblib
files from a trusted source; the web interface never accepts uploaded models.
