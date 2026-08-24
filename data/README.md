# Data

`processed/demo_application_development.csv` and
`processed/demo_application_queue.csv` are synthetic chemical-level inputs for
the fixed EcoOOD application protocol described in the repository README. They
contain no provider records and can be used to verify signal selection and
four-route screening output.

This directory contains the small public demo input used by EcoOOD examples and
tests, together with the field-level manifest for the current application
configuration. The original benchmark snapshot, its feature manifest, split
assignments, and compact prediction outputs are available in the
[v0.2.1 analysis release](https://github.com/bubaizhanshen/EcoOOD/releases/tag/v0.2.1).

## Layout

- `processed/demo_ecoood.csv`: synthetic benchmark example
- `processed/echa_external_main.csv`: fixed 100-case, 48-chemical ECHA
  freshwater evaluation set
- `processed/echa_external_seven_species.csv`: nested 93-case, 47-chemical
  seven-species sensitivity set
- `processed/echa_extension_candidates.csv`: deterministic 150-chemical ECHA
  extension sample selected from chemical identity and endpoint availability
  before toxicity values or model predictions were examined
- `feature_manifest.csv`: field names, roles, cardinalities, and missingness
  rates after applying the current molecular-input and feature rules (4,611
  records; 801 chemicals); publication year and the bioactivity coverage count
  are audit-only fields
- `raw/`: provider downloads created by the source-specific fetch scripts
- `processed/`: locally built benchmark tables and feature caches

The original 4,942-record snapshot and its exact manifest remain versioned in
the analysis release so the published benchmark results can be reproduced.
The two external evaluation tables contain derived quantitative records used in
the manuscript and retain their ECHA source identifiers. They contain no raw
dossier pages. Both are identity-disjoint from the benchmark by CASRN and
InChIKey; the seven-species table is a nested subset of the main table.
External case targets are medians on the `log10(mol L-1)` scale within
chemical-species-endpoint groups. `molar_concentration` and `toxicity_value`
are the back-transformed case target, and `toxicity_unit` is `M`; the original
source value and unit remain available only in the local row-level extraction
audit.
The full processed tables can also be rebuilt from local ECOTOX,
DSSTox/CompTox, invitrodb/ToxCast, ECHA, and PMRA source files with the commands
in the main [README](../README.md).

## Local Files

Typical local paths are:

- `data/raw/`
- full `data/processed/ecotox_acute_ecoood_*` benchmark tables
- `data/processed/invitrodb_mechanism_features*.csv` (historical file name;
  used as a bioactivity-proxy cache)
- local PubChem/CompTox cache expansions
- partial downloads such as `*.part`
