# AquaTriage Review Workspace

## What the user receives

A reproducible list of chemicals selected for follow-up, accompanied by predicted
acute effect concentrations, the ranking measure, endpoint coverage, missing
inputs, and nearby training cases. Review means checking or measuring a chemical;
selection alone does not change its toxicity prediction.

The workspace implements the two primary review tasks:

| Task | Review population | Preset | User setting |
| --- | --- | --- | --- |
| Check prediction accuracy | All eligible chemicals | Block-normalized kNN | Review count or percentage |
| Find potentially missed hazards | Predicted-low-concern chemicals first | Threshold proximity | Concentration criterion and review count or percentage |

Choose an objective and review count or percentage, then run the review. The
objective preset is selected automatically. To compare another measure, open
**Ranking options**, turn off **Use objective preset**, and select the measure.
The concentration criterion is shown only for hazard-omission review.

Similarity AD and raw ensemble SD can also rank the same queues. The workspace
does not choose a winner using unknown outcomes, average these measures into a
new score, or assign binary reliability labels. The presets implement the
study's task distinction; a different application still needs relevant
validation.

## Inputs

Choose the example queue, enter individual cases, or upload a CSV. The entry
form collects a confirmed structure and identifier, a species, an endpoint,
and the actual exposure duration. Species can be selected from the reference
or entered by scientific name. Each added case uses the same checks as a CSV
row. Optional test conditions can be supplied in the form; the full CSV template
also accepts taxonomic ranks and bioactivity measurements.

Before running, the workspace reports eligible chemicals, rejected cases, and
the resulting number of review places. Unrecognized columns are identified
and ignored. The Input checks tab provides rejection reasons and a downloadable
issues table.

One CSV row represents one chemical, species, endpoint, and set of test conditions.
Repeated records are allowed; a chemical ID receives only one review place.
Identifiers are read as text, preserving leading zeros.

| Column | Meaning | Required |
| --- | --- | --- |
| `chemical_id` | Stable identifier for the actual test substance | Yes |
| `smiles` | Confirmed fixed-composition structure, including the relevant substance form | Yes |
| `identity_verified` | `true` or `1` attests that the identity-structure pair has been checked | Yes |
| `species` | Scientific species name | Yes |
| `endpoint` | One of the endpoint codes below | Yes |
| `duration_h` | Actual planned exposure duration, in hours | Yes |
| `chemical_name`, `case_id` | Display name and optional unique case identifier | No |
| `genus`, `family`, `order`, `class_name`, `phylum` | Taxonomic ranks | No |
| `medium` | `FW`, `FW/`, `AQU`, `AQU/`, `CUL`, or `CUL/` | No |
| `temperature_c`, `ph`, `ctx_hardness` | Temperature (degrees C), pH, and hardness in the fitted reference's units | No |
| 34 `mech_*` variables | Assay summaries specified in the bioactivity manifest | No |

| Endpoint code | Biological response | Accepted duration |
| --- | --- | --- |
| `fish_96h_lc50` | Fish median lethal concentration | 90-102 h |
| `daphnia_48h_ec50` | Daphnid median effective concentration | 42-54 h |
| `algae_72_96h_ec50` | Algal median effective concentration | 66-102 h |

Endpoint and duration remain separate inputs. An endpoint label is not a reason
to replace an actual duration with a nominal value.

The software checks representation eligibility, not whether a name truly
corresponds to a supplied SMILES. Unconfirmed identities are rejected. A SMILES
must be parseable, carbon-containing, and fixed-composition, without wildcard
atoms or repeating-polymer definitions. Generic salt names require identity
review; confirmed single-molecule inner salts are supported. Explicit
fixed-composition structures retain their counterions in descriptor and mass
calculations. Do not submit a parent structure with a salt's identity.

One ID associated with different structures is rejected. Different confirmed
IDs sharing a molecular representation are flagged, not silently merged:
stereounspecified SMILES may represent distinct verified stereoisomers.

Missing optional inputs use training-partition imputation. Taxonomy is first
filled from an unambiguous species lookup in the fitted reference, with the
number of filled values exported. New species are flagged. No online lookup
or invented assay measurement is used. Consult
[the bioactivity manifest](../data/bioactivity_feature_manifest.csv) for units
and definitions; values with the wrong units must not be pasted into these
columns.

The browser does not convert hardness units. Leave hardness blank unless its
units and concentration basis match the reference; record that convention when
building a reference from your own data.

## Fitting a reference

The local browser's **Fit from my data** option and the CLI use the same backend.
The public instance uses the installed reference and does not accept training jobs.
The measured reference must include either:

- `target_log_molar`: measured log10(mol/L); or
- `toxicity_value` and `toxicity_unit`: an uncensored positive concentration.

When `target_log_molar` is present it is authoritative. Supported concentration
units include mol/L, mM, micromolar (`uM`), nM, g/L, mg/L, ug/L, ng/L, and
ppm/ppb/ppt for dilute aqueous tests. An `AI ` prefix denotes concentration of
the active ingredient; it does not perform a formulation correction.
Mass concentration is converted with the resolved structure's molecular weight.
Nonpositive, explicitly censored, unsupported, and nonfinite targets are rejected.
The application stops on invalid training records instead of silently removing
them.

The application computes all seven physicochemical descriptors with RDKit,
including molecular weight and MolLogP. It fits preprocessing and five bootstrap
LightGBM members on the supplied reference, with 400 trees per member and the
study's fixed tree settings. Publication year, source identifiers, targets, and
unrecognized columns cannot become predictors. Optional all-missing numeric
columns retain the feature builder's zero fallback; missingness is still
reported. The minimum of 30 records and 10 chemicals is a software guard,
not evidence that such a small reference is scientifically sufficient.

An application fit uses the supplied reference as training data. It is not a
new cross-validation experiment and does not inherit the paper's held-out
performance numbers. The local demonstration reference was fitted on the
4519-record, 775-chemical corrected benchmark. A trained artifact contains
model objects, transformations, and training-reference features, so it should
only be loaded from a trusted source. Uploaded pickle/joblib models are not
accepted by the web interface.

## Calculations

1. Each query is transformed with the fitted reference's imputation, scaling,
   categorical encoding, and 2048-bit radius-2 Morgan fingerprint.
2. The mean of five model predictions is the log10 molar point estimate. Their
   population standard deviation is the raw ensemble-SD comparator.
3. Similarity AD is one minus the maximum training-set Tanimoto similarity.
   Block kNN divides each feature block by its training RMS row norm and the
   square root of the number of blocks, then averages Euclidean distances to
   five training cases. These definitions reuse the benchmark implementation.
4. Scores and log concentrations are median-aggregated within chemical-endpoint
   groups. The lowest endpoint concentration and largest endpoint support/SD
   score summarize each chemical. Missing endpoints are not assigned values.
5. For omission review, a chemical is predicted high concern if any represented
   endpoint median is at or below the supplied mass-concentration criterion.
   The proximity rule sorts the other chemicals by increasing log concentration
   margin. Other rules sort by decreasing score. Predicted-high-concern chemicals
   only enter after the low-concern queue is exhausted.
6. Percentage workloads select `round(fraction * N)` chemicals, with Python's
   nearest-even convention at exact halves. `N` is all eligible chemicals.
   Count workloads use the stated integer directly. Chemical IDs resolve ties.

The 1 mg/L value is the study's acute-effect concentration reference, not a
complete GHS classification or an exposure-based risk assessment.
Queue measurements, if accidentally included, are ignored for prediction and
ranking. Rejected cases are reported separately and do not consume the quota.
Partially rejected chemicals retain an explicit rejected-case count.

## Results

After a successful run, the workspace opens the review queue with selected
chemicals shown first. Turn off **Selected only** to include the rest of the
queue. Selecting a table row opens that chemical's record; the arrow buttons
move through the full review order, and **Queue** returns to the filtered list.

The queue displays the active ranking measure. For omission review, it also
shows the predicted concentration relative to the criterion and the
prediction/criterion ratio. The proximity score is the base-10 logarithm of
this ratio; smaller positive margins receive priority in the predicted-low-concern
queue. Other ranking measures remain available in the expandable scores table.

Each chemical's detail view includes its molecular structure, endpoint-specific
concentrations, the follow-up question, the numerical ranking basis, case
results, and training neighbors. The endpoint chart uses
a logarithmic mg/L axis and shows the criterion only for omission review.
An unrepresented endpoint is marked as having no eligible input case, not
assigned a concentration. Nearest neighbors are training cases in the full
normalized input space, not necessarily distinct chemicals or structural analogs.

The selected-chemicals CSV contains the full selected queue, regardless of the
on-screen search filter, including each chemical's follow-up question and
ranking basis. A separate text summary records the completed task,
ranking rule, quota, units, and reference/input identifiers. Changing the input
or settings marks the previous results as outdated and disables result downloads
until the review is rerun.

The downloaded ZIP contains:

| File | Content |
| --- | --- |
| `chemicals.csv` | Ranked chemical queue, selected flag, reason, scores, coverage |
| `endpoints.csv` | Chemical-endpoint medians |
| `cases.csv` | Individual predictions, scores, missingness and overlap flags |
| `neighbors.csv` | Five nearest reference cases and distances for each query |
| `rejected.csv` | Input rows that did not pass the checks and their reasons |
| `run_manifest.json` | Input/reference hashes, protocol, settings, quota and aggregation |

Prediction concentrations are exported in both log10(mol/L) and mg/L.
The mg/L chemical summary is the minimum endpoint median, not a measured value.
Unselected means **not selected at the chosen workload**, not safe or reliable.
Spreadsheet-formula-like text is escaped in downloadable CSVs.

## Launch and deployment

```bash
python -m pip install -r app/requirements.txt
python scripts/serve_review.py --port 8501
```

The server binds to `127.0.0.1`. For a remote research workstation, forward the
port over SSH rather than exposing an unauthenticated upload endpoint:

```bash
ssh -L 8501:127.0.0.1:8501 USER@HOST
```

Runtime caches remain in the project work directory. Uploads are held in the
server session and are not sent to third-party services. Downloads are explicit.
Use access control and a deployment review before making an instance public.
The default local model is bundled under `models/review-v1/`. Use
`--reference path/to/reference.joblib` to choose your own fitted model.
The inference-only public entrypoint is `app/cloud.py`; see
[cloud deployment](cloud_deployment.md) for account setup and publishing.

## Code map

| Responsibility | File |
| --- | --- |
| Input contract, prediction, aggregation, ranking, export | `src/ecoood/review_workflow.py` |
| Result labels, endpoint display data, and run summaries | `src/ecoood/review_presentation.py` |
| Fitted preprocessing and Morgan fingerprints | `src/ecoood/features.py` |
| Bootstrap prediction models | `src/ecoood/models.py` |
| Block normalization and Tanimoto novelty | `src/ecoood/ad.py` |
| Molecular representation checks | `src/ecoood/molecular_identity.py` |
| Browser workspace | `app/review.py` |
| Local server launcher | `scripts/serve_review.py` |
| Batch fit/review commands | `scripts/run_review_workflow.py` |
| Contract and numerical tests | `tests/test_review_workflow.py` |
