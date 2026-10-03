"""Export units, released source columns, and transformations for assay predictors."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ecoood.invitrodb import H295R_HORMONE_COLS, LITERATURE_MODE_MAP

SOURCE = "https://doi.org/10.23645/epacomptox.6062623"


def definitions() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}

    def add(name: str, block: str, source_column: str, units: str, operation: str, meaning: str):
        rows[name] = dict(source_group=block, source_column=source_column,
                          units=units, aggregation=operation, definition=meaning,
                          source_url=SOURCE, code="src/ecoood/invitrodb.py")

    for key, source, unit, meaning in [
        ("median_um", "cytotox_median_um", "micromol/L", "Median cytotoxicity-burst concentration"),
        ("lower_um", "cytotox_lower_bound_um", "micromol/L", "Lower cytotoxicity-burst concentration bound"),
        ("median_log10_um", "cytotox_median_log", "log10(micromol/L)", "Released log-scale cytotoxicity median"),
        ("lower_log10_um", "cytotox_lower_bound_log", "log10(micromol/L)", "Released log-scale cytotoxicity lower bound"),
        ("ntested", "ntested", "count", "Number of tested assays in the released cytotoxicity summary"),
        ("nhit", "nhit", "count", "Number of hit assays in the released cytotoxicity summary"),
    ]:
        add("mech_cytotox_" + key, "Cytotoxicity summary", source, unit,
            "Retain released chemical-level value", meaning)
    add("mech_cytotox_hit_rate", "Cytotoxicity summary", "nhit; ntested", "fraction",
        "nhit/ntested; missing if ntested=0", "Fraction of tested assays reported as hits")
    for mode, name in LITERATURE_MODE_MAP.items():
        add(name, "AR/ER literature summary", "literature_mode; literature_score", "ordinal score (0 to 1)",
            "inactive=0; very weak=0.25; weak=0.5; moderate/medium=0.75; strong=1; chemical maximum",
            mode.replace("_", " ") + "; qualitative source category, not a measured concentration")
    for receptor in ["ar", "er"]:
        for action in ["agonist", "antagonist"]:
            add(f"mech_toxcast_{receptor}_auc_{action}", "ToxCast AR/ER pathway summary",
                "auc_" + action, "dimensionless model score", "Retain released chemical-level value",
                f"{receptor.upper()} {action} pathway area-under-curve score")
    for hormone in H295R_HORMONE_COLS:
        add(f"mech_h295r_{hormone.lower()}_max_abs", "HT-H295R model summary", hormone,
            "released hormone-response scale", "Maximum absolute released response across rows of one DTXSID",
            f"{hormone} response summary; source scale retained without converting to a hormone concentration")
    for key, source, unit, operation, meaning in [
        ("mean_abs", "; ".join(H295R_HORMONE_COLS), "released hormone-response scale",
         "Mean absolute response across all observed hormone columns and rows of one DTXSID", "Mean magnitude of released hormone responses"),
        ("bmd_min", "BMD", "micromol/L", "Minimum released BMD across rows of one DTXSID", "Concentration at which the fitted pathway response reaches its critical value"),
        ("mmd_max", "mMd", "dimensionless", "Maximum released mMd across rows of one DTXSID", "Mean Mahalanobis distance of the multivariate hormone response"),
        ("maxmmd_max", "maxmMd", "dimensionless", "Maximum released maxmMd across rows of one DTXSID", "Released maximum mean Mahalanobis response"),
        ("critical_val", "criticalVal", "dimensionless", "Median released criticalVal across rows of one DTXSID", "Released critical value for the multivariate pathway response"),
        ("active_endpoint_count", "; ".join(H295R_HORMONE_COLS), "count",
         "Count of 11 chemical-level maximum absolute hormone summaries >=1", "Engineered response-count descriptor; not a validated endocrine activity classification"),
    ]:
        add("mech_h295r_" + key, "HT-H295R model summary", source, unit, operation, meaning)
    assert len(rows) == 34
    return rows


def build_manifest(data: pd.DataFrame) -> pd.DataFrame:
    definitions_by_field = definitions()
    fields = {c for c in data if c.startswith("mech_") and c != "mech_feature_count"}
    if fields != set(definitions_by_field):
        raise ValueError(f"Unexpected bioactivity schema: {fields ^ set(definitions_by_field)}")
    chemical = data.groupby("chemical_id")[sorted(fields)].first()
    result = []
    for field, definition in sorted(definitions_by_field.items()):
        result.append(dict(field=field, **definition,
            observed_records=int(data[field].notna().sum()),
            observed_chemicals=int(chemical[field].notna().sum()),
            missing_fraction=float(data[field].isna().mean()),
            predictor_processing="Training-median imputation and training-standardization",
            reliability_processing="Jointly observed standardized values; missingness scored separately"))
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_manifest(pd.read_csv(args.benchmark)).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
