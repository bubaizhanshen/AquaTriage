# AquaTriage

[![Tests](https://github.com/bubaizhanshen/AquaTriage/actions/workflows/tests.yml/badge.svg)](https://github.com/bubaizhanshen/AquaTriage/actions/workflows/tests.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

AquaTriage is a Python workflow for reliability assessment and review
prioritization in aquatic ecotoxicity screening. It evaluates
chemical-species-context cases before and after toxicity prediction and
measures how often reliability signals place measured high-concern chemicals
predicted as lower concern into review under a fixed review budget.
The Python package remains importable as `ecoood`.

AquaTriage is the full assessment framework. The **calibration-trained
prediction-error risk score** is one component of that framework: a logistic
high-error ranking score fitted to calibration-fold residual labels. It is used
for ranking and is not a calibrated error
probability. The benchmark compares it with structural similarity,
input-space distance, ensemble uncertainty, and comparison models trained on
the same calibration residual labels.

## Workflow

1. Route unresolved identities, mixtures, and unsupported molecular
   representations directly to `withhold/review`.
2. Fit the toxicity predictor and reliability models using separate training
   and calibration partitions.
3. Quantify chemical, biological, contextual, bioactivity-proxy, and ensemble
   uncertainty signals for scoreable cases.
4. Rank predicted-low-concern chemicals by threshold proximity for the primary
   omission-control objective; use a labeled development subset to compare
   predefined reliability measures when that objective and data are available.
5. Assign predictions to `screen now`, `lower priority`, `withhold/review`, or
   `prioritize testing`.

The current application predictor excludes ECOTOX reference publication year
from model and reliability inputs; year is retained only for chronological
splitting and audit. RDKit 2024.03.2 MolLogP is recomputed from every eligible
structure so logP availability does not depend on external database coverage.
The invitrodb/ToxCast feature-count field is retained for coverage audits but
does not enter the predictor or the bioactivity-support distance.
The benchmark also reports a distinct-chemical kNN sensitivity in which each
training chemical contributes at most one of the five nearest support cases.
The standard case-level block-normalized kNN distance is a support-ranking
candidate; the distinct-chemical version tests whether repeated records
from frequently studied chemicals materially change the ranking.

The current application prediction-error risk score uses eight predefined
signals. Chemical support is represented by a five-neighbor Tanimoto distance
over distinct training chemicals and a shrinkage Mahalanobis descriptor
distance. Biological support uses taxonomic distance based on the deepest
training-supported lineage rank. Test context is
compared with a same-endpoint, missing-aware Gower distance and an explicit
context-missingness fraction. Bioactivity-proxy support uses a missing-aware
standardized distance over distinct chemicals and its missingness fraction.
The final signal is bootstrap-ensemble SD. The original distance definitions,
cosine fingerprint distance, and 1, 3, 5, and 10 neighbors are included in
sensitivity analyses.

## Repository Layout

```text
configs/                Fixed benchmark configurations
data/                   Synthetic benchmark input and predictor field manifest
scripts/                Data preparation, benchmark, and audit entry points
src/ecoood/             Python package
tests/                  Unit and regression tests
```

Generated benchmark outputs are stored locally under `outputs/`. The current
strict-input benchmark, frozen cutoffs, case-level predictions, and compact
analysis tables are distributed through the
[v0.3.0 analysis release](https://github.com/bubaizhanshen/AquaTriage/releases/tag/v0.3.0).
The processed external evaluation panels used in the current manuscript are
included under `data/processed/` because they are small, fixed analysis inputs;
the earlier ECHA-only tables are retained for historical compatibility.

## Installation

Create the reference environment:

```bash
conda create -y -n ecoood python=3.11 pip
conda install -y -n ecoood -c conda-forge \
  pandas scikit-learn scipy pyarrow pyyaml requests joblib beautifulsoup4 \
  lightgbm xgboost openpyxl rdkit=2024.03.2
conda run -n ecoood python -m pip install -e . --no-deps
```

The exact package versions used for the reference analyses are recorded in
`requirements-reproducibility.txt`.

For development:

```bash
conda run -n ecoood python -m pip install -e ".[dev]"
```

Run the tests:

```bash
conda run -n ecoood pytest -q
```

## Apply AquaTriage to a New Screening Queue

AquaTriage application uses a fixed candidate library, selection rule, and four
screening routes. A screening program supplies its concern rule, review fraction,
review objective, and any available measurements from the intended screening setting.

The command defaults to `low_concern_first` for `false_negative_capture`:
predicted-low-concern chemicals fill the review quota before predicted-high-concern
chemicals. Other objectives default to `all_queue`. Set `--review-policy` explicitly
to reproduce the intended analysis. The Python functions retain `all_queue` as a
backward-compatible default; pass `review_policy="low_concern_first"` for the
primary omission-control analysis.

The review fraction accepts values from 0 to 1, inclusive. The queue selects
`round(review_fraction * N)` eligible chemicals, using Python's nearest-even
rounding at exact halves. Zero review selects none; full review selects all.
Ineligible inputs remain in `withhold/review` outside this review allocation.

The application input is a chemical-level scored table. Case-level predictions
must first be median-aggregated within chemical-endpoint groups and then combined
across endpoints. The queue must contain:

- one unique `chemical_id` per row;
- `pred_high_concern`, calculated from a predefined endpoint or program concern
  rule;
- `input_eligible`, indicating whether identity and molecular input checks passed;
- the score columns in the fixed candidate library for chemicals that pass
  the input eligibility gate. Scores may be empty for ineligible chemicals.

The primary policy additionally requires `prediction_distance` (threshold proximity)
and `interval_cross_score`. These are chemical-level scores produced from
endpoint-specific predictions and frozen concern cutoffs by
`scripts/analyze_stage1_decision_baselines.py`: proximity is the maximum negative
absolute prediction-to-cutoff distance across endpoints; interval crossing is
positive for a crossing and increases with its depth. Do not calculate these
cutoffs from the evaluation outcomes. Use `--candidate NAME=COLUMN` to explicitly
supply a smaller, predefined candidate set. Candidate selection uses the same
review policy as routing, and development/evaluation chemical IDs must be disjoint.

The optional development table contains the same columns plus
`true_high_concern`. For error-ranking objectives it also contains `y_true` and
`y_pred`. Directional-underprediction selection uses a nonnegative
`directional_underprediction_loss` per chemical. Higher candidate scores always
indicate greater priority for review.

Run the included synthetic reliability-only example (all-queue illustration):

```bash
conda run -n ecoood python scripts/apply_ecoood.py \
  --development data/processed/demo_application_development.csv \
  --queue data/processed/demo_application_queue.csv \
  --objective false_negative_capture \
  --review-policy all_queue \
  --review-fraction 0.25 \
  --output-dir outputs/demo_application
```

The command writes:

- `signal_selection.csv`, containing the fixed-workload comparison and selected
  measure;
- `routed_queue.csv`, containing the selected score, review rank, route, and route
  reason for every chemical;
- `application_summary.json`, recording the objective, selected measure, review
  fraction, and route counts.

For `false_negative_capture`, the fixed rule first maximizes the fraction of
measured high-concern chemicals rescued from a predicted lower-priority queue;
the fraction of high-concern chemicals left at low priority and the
lower-priority false-omission rate resolve ties. The other supported objectives
are `directional_underprediction_area`, `largest_error_capture`, and
`overall_error_ranking`. The directional objective selects the measure with the
largest area under the cumulative capture curve for toxicity-underprediction
loss across the full 0-100% development-set review range.

Without development labels, the primary low-concern-first command uses
`threshold_proximity`, the task-aligned baseline with the highest mean capture
in the current internal and external comparisons. This is a baseline allocation,
not a newly validated adaptive selector. `all_queue` retains
`block_normalized_knn` as its default support ranking. Similarity AD is retained
as a separate structural novelty assessment. Use `--default-signal similarity_ad` when structural novelty
is the review objective; new identities or scaffolds alone do not establish
which measure best reduces false omissions. A distinct-chemical-neighbor
version is retained as a sensitivity analysis to test whether repeated records
from frequently studied chemicals affect the ranking. The output retains all
candidate scores so the evidence record remains auditable. Candidate names and
columns can be overridden explicitly with repeated `--candidate NAME=COLUMN`
arguments.

## Quick Start

The included synthetic table supports a short smoke test:

```bash
conda run -n ecoood python scripts/run_benchmark.py \
  --data data/processed/demo_ecoood.csv \
  --splits random scaffold temporal species chemical_class \
  --models random_forest \
  --members 2 \
  --output-dir outputs/demo_benchmark
```

## Reproduce the Primary Benchmark

The primary benchmark is the finalized strict-input table containing 4,611
records from 801 chemicals. Its SHA-256 checksum is:

```text
273be0b1c3e34f7a9d9665675049b63f6cd5ae05393027487a98945388074c42
```

The primary benchmark uses RDKit 2024.03.2. Scaffold generation can differ
between RDKit releases, so reproducing the primary scaffold partition requires
this version.

Audit the predictor fields, duplicate records, structure parsing, and current
molecular-input eligibility rule:

```bash
conda run -n ecoood python scripts/audit_benchmark_integrity.py \
  --data path/to/EcoOOD_benchmark_snapshot_structured.csv \
  --output-dir outputs/integrity_audit \
  --strict-eligible-output outputs/integrity_audit/strict_input_benchmark.csv
```

The finalized rule retains 4,611 records from 801 chemicals. Under RDKit
2024.03.2, it reclassifies 326 records with carbon-free structures and five
records with unparseable structures to `withhold/review`. All primary benchmark
settings are fitted on this strict table.

The five unparseable records in the original benchmark can be reproduced only with the
explicit `--allow-legacy-structure-placeholder` flag. The flag is disabled by
default and is not used by the current application workflow.

Run the fixed seed panel on the audited molecular-input table:

```bash
conda run -n ecoood python scripts/run_integrity_benchmark.py \
  --data outputs/integrity_audit/strict_input_benchmark.csv \
  --output-root outputs/integrity_benchmark
```

Summarize fixed-workload outcomes:

```bash
conda run -n ecoood python scripts/summarize_multiseed_screening.py \
  --input-root outputs/integrity_benchmark \
  --output-dir outputs/integrity_benchmark/aggregate
```

Generate the statistical audit tables, including paired comparisons with
random review and the 15–35% fixed-workload sensitivity:

```bash
conda run -n ecoood python scripts/summarize_benchmark_audits.py \
  --benchmark-root outputs/integrity_benchmark \
  --output-dir outputs/integrity_benchmark/analysis_tables \
  --bootstrap-replicates 2000
```

Generate the complete 0–100% review-workload curves:

```bash
conda run -n ecoood python scripts/analyze_full_review_workload.py \
  --panel outputs/integrity_benchmark/aggregate/chemical_split_panel_endpoint_relative.csv \
  --output-dir outputs/integrity_benchmark/full_workload
```

Reproduce the fixed 1, 10, and 100 mg/L acute-effect screening comparisons
from the inputs in the v0.3.0 release archive:

```bash
conda run -n ecoood python scripts/analyze_absolute_hazard_thresholds.py \
  --benchmark data/strict_input_benchmark.csv \
  --internal-predictions predictions/internal_predictions.csv \
  --cutoffs data/calibration_cutoffs.csv \
  --external-predictions predictions/external_predictions.csv \
  --external-panel data/external_panel.csv \
  --output-dir outputs/fixed_hazard
```

Extract the release archive into the repository root before running this command.
The script uses frozen model predictions and compares each ranking with random
review of the same predicted-low-concern queue at an equal workload.

Run input-feature, molecular-parser, and leakage controls:

```bash
conda run -n ecoood python scripts/run_integrity_sensitivity.py \
  --data path/to/EcoOOD_benchmark_snapshot_structured.csv \
  --output-dir outputs/integrity_sensitivity
```

Evaluate endpoint-specific concern cutoffs fixed from calibration partitions:

```bash
conda run -n ecoood python scripts/analyze_calibration_frozen_thresholds.py \
  --data path/to/EcoOOD_benchmark_snapshot_structured.csv \
  --prediction-root outputs/integrity_benchmark \
  --output-dir outputs/calibration_cutoff_sensitivity
```

Reproduce the sensitivity analysis that motivated excluding reference
publication year from predictor and reliability inputs:

```bash
conda run -n ecoood python scripts/analyze_temporal_without_publication_year.py \
  --data path/to/EcoOOD_benchmark_snapshot_structured.csv \
  --output-dir outputs/temporal_without_publication_year
```

Additional audit entry points cover exact duplicates, reference holdout,
fixed temporal transfer, species-chemical overlap, configuration sensitivity,
local interval recalibration, and external dossier transfer. Run any script
with `--help` to inspect its inputs and output layout.

The chemical-class holdout uses five fixed leave-one-class-out folds:
PFAS, conazoles, neonicotinoids, PPCPs, and strobins. A chemical carrying
multiple class labels enters the test fold whenever any label matches the
held-out class; the same class label is therefore absent from training and
calibration.

## Data Sources

The benchmark is derived from public ECOTOX, DSSTox/CompTox, and
invitrodb/ToxCast resources. The external transfer audit uses Japanese Ministry
of the Environment summaries and ECHA/PMRA records. Provider downloads are
transformed locally with the scripts in this repository; the release archive
contains the strict benchmark, fixed cutoffs, predictions, analysis tables, and
file checksums.

The predictor field names, roles, cardinalities, and missingness rates are
listed in [`data/feature_manifest.csv`](data/feature_manifest.csv). Target
fields, identifiers, source-document metadata, and chemical-class labels are
excluded from the predictor matrix. Bioactivity fields are treated as partial
in vitro proxies.

To rebuild the structured table from local provider files:

```bash
conda run -n ecoood python scripts/build_ecotox_dataset.py \
  --structure-cache data/raw/dsstox_priority_cache_1000.csv \
  --dsstox-source data/raw/DSSTox_CCD_dump_12092025_CSVs.zip \
  --mechanism-cache data/processed/invitrodb_mechanism_features_v43.csv \
  --invitrodb-summary data/raw/INVITRODB_SUMMARY.zip \
  --max-chemicals 1000 \
  --output data/processed/ecotox_acute_ecoood_1000chem_dsstox_mech.csv \
  --structured-output data/processed/ecotox_acute_ecoood_1000chem_dsstox_mech_structured.csv
```

The historical cache filename contains `mechanism`; the current analysis uses
these fields as bioactivity proxies.

## External Transfer Audit

The repository retains the earlier ECHA-only tables for compatibility and also
includes the current expanded external panel:

- `data/processed/ecoood_external_expanded_v1.csv`: 1,027 freshwater cases
  across 441 chemicals, derived from the retained historical ECHA/REACH cohort
  and official Japanese Ministry of the Environment summaries;
- `data/processed/echa_external_main.csv`: historical 100-case, 48-chemical
  ECHA-only panel;
- `data/processed/echa_external_seven_species.csv`: historical nested sensitivity
  set with 93 cases across 47 chemicals;
- `data/processed/echa_extension_candidates.csv`: identity-only extension sample
  selected before toxicity values or model predictions were inspected.

The expanded panel is identity-disjoint from the benchmark and between its two
source cohorts under CASRN, InChIKey, structure-connectivity, and parent-family
checks. Refit the current application model and score it with:

```bash
conda run -n ecoood python scripts/run_echa_pmra_external_validation.py \
  --data-path outputs/integrity_audit/strict_input_benchmark.csv \
  --panel-path data/processed/ecoood_external_expanded_v1.csv \
  --output-dir outputs/external_expanded_v1

conda run -n ecoood python scripts/analyze_external_false_reassurance.py \
  --train-path outputs/integrity_audit/strict_input_benchmark.csv \
  --predictions outputs/external_expanded_v1/external_predictions_all.csv \
  --output-dir outputs/external_expanded_v1/false_reassurance
```

Simulate target-domain measure selection before the remaining external queue is
evaluated. Development and evaluation sets are disjoint at chemical level. The
measure that best ranks directional toxicity-underprediction loss over the full
development-set workload curve is fixed before evaluation outcomes are used:

```bash
conda run -n ecoood python scripts/analyze_external_deployment_selection.py \
  --train-path outputs/integrity_audit/strict_input_benchmark.csv \
  --predictions outputs/external_expanded_v1/external_predictions_all.csv \
  --development-size 20 \
  --repeats 100 \
  --review-fraction 0.25 \
  --output-dir outputs/external_expanded_v1/deployment_selection
```

The outputs include the selected measure for every repetition, comparisons with
fixed similarity AD, fixed block-normalized kNN, prediction-error risk, and
random review, plus the four screening routes assigned to each evaluation
chemical.

To reconstruct the extension sample from the live eChemPortal/ECHA endpoint
index, use only chemical identity and endpoint availability:

```bash
conda run -n ecoood python scripts/select_echa_external_candidates.py \
  --benchmark outputs/integrity_audit/strict_input_benchmark.csv \
  --existing-external path/to/corrected_pmra_selected_external.csv \
  --candidate-limit 150 \
  --output-dir outputs/echa_extension_selection
```

The selected identities can then be passed to
`build_echa_pmra_external_rows.py`, `prepare_echa_pmra_validation_panel.py`, and
`build_echa_pmra_clean_panel.py`. These stages parse exact endpoint-duration
records, preserve censoring operators, resolve molecular inputs, and retain only
supported quantitative freshwater experimental studies. Raw dossier pages and
provider archives are cached locally under `outputs/` or `data/raw/` and are not
committed. External uncertainty is resampled by chemical identity.

## License

Released under the [MIT License](LICENSE).
