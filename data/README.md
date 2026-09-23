# Data

`processed/demo_application_development.csv` and
`processed/demo_application_queue.csv` are synthetic chemical-level inputs for
the fixed EcoOOD application protocol described in the repository README. They
contain no provider records and can be used to verify signal selection and
four-route screening output.

This directory contains the small public demo input used by AquaTriage examples and
tests, together with the field-level manifest for the current application
configuration. The strict benchmark, fixed cutoffs, case-level predictions,
and compact analysis tables are available in the
[v0.3.0 analysis release](https://github.com/bubaizhanshen/AquaTriage/releases/tag/v0.3.0).

## Layout

- `processed/demo_ecoood.csv`: synthetic benchmark example
- `processed/ecoood_external_expanded_v1.csv`: current 1,027-case,
  441-chemical identity-disjoint external panel derived from the retained
  historical ECHA/REACH cohort and official Japanese Ministry of the
  Environment summaries
- `processed/echa_external_main.csv`: historical 100-case, 48-chemical ECHA
  freshwater evaluation set
- `processed/echa_external_seven_species.csv`: historical nested 93-case,
  47-chemical seven-species sensitivity set
- `processed/echa_extension_candidates.csv`: deterministic 150-chemical ECHA
  extension sample selected from chemical identity and endpoint availability
  before toxicity values or model predictions were examined
- `feature_manifest.csv`: field names, roles, cardinalities, and missingness
  rates after applying the current molecular-input and feature rules (4,611
  records; 801 chemicals); publication year and the bioactivity coverage count
  are audit-only fields
- `raw/`: provider downloads created by the source-specific fetch scripts
- `processed/`: locally built benchmark tables and feature caches

The primary prediction table contains 4,611 records from 801 chemicals under
the strict molecular-input configuration described in the feature manifest.
The expanded external panel contains derived quantitative records used in the
current manuscript analysis and retains source identifiers plus available source
URLs. It contains no raw dossier pages. The expanded panel is identity-disjoint from the
benchmark and between its two source cohorts by CASRN, InChIKey, structure
connectivity, and parent-family checks. The historical ECHA tables remain
available for compatibility; the seven-species table is a nested subset of the
historical main table.
External case targets are medians on the `log10(mol L-1)` scale within
chemical-species-endpoint groups. `molar_concentration` is the back-transformed
case target; `toxicity_value` and `toxicity_unit` preserve the source-scale value
and unit where available. The original source value and unit remain available in
the local row-level extraction audit.
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
