from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine identity-disjoint ECHA evaluation subsets."
    )
    parser.add_argument("--policy-panel", type=Path, required=True)
    parser.add_argument("--extension-panel", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized_values(series: pd.Series) -> set[str]:
    return {
        str(value).strip().upper()
        for value in series.dropna()
        if str(value).strip()
    }


def main() -> None:
    args = parse_args()
    policy = pd.read_csv(args.policy_panel).assign(external_subset="pmra_selected")
    extension = pd.read_csv(args.extension_panel).assign(external_subset="echa_extension")
    benchmark = pd.read_csv(args.benchmark)

    policy_cas = normalized_values(policy["casrn"])
    extension_cas = normalized_values(extension["casrn"])
    if overlap := policy_cas & extension_cas:
        raise ValueError(f"External subsets overlap by CASRN: {sorted(overlap)}")

    combined = pd.concat([policy, extension], ignore_index=True, sort=False)
    benchmark_cas = normalized_values(benchmark["casrn"])
    benchmark_keys = normalized_values(benchmark["inchikey"])
    combined_cas = normalized_values(combined["casrn"])
    combined_keys = normalized_values(combined["inchikey"])
    if overlap := combined_cas & benchmark_cas:
        raise ValueError(f"External set overlaps benchmark by CASRN: {sorted(overlap)}")
    if overlap := combined_keys & benchmark_keys:
        raise ValueError(f"External set overlaps benchmark by InChIKey: {sorted(overlap)}")
    if combined["inchikey"].isna().any():
        raise ValueError("Every external chemical must have a resolved InChIKey.")

    duplicated = combined.duplicated(["chemical_id", "endpoint", "species"], keep=False)
    if duplicated.any():
        cases = combined.loc[duplicated, ["chemical_id", "endpoint", "species"]]
        raise ValueError(f"Duplicate external cases remain:\n{cases.to_string(index=False)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(
        f"Wrote {len(combined)} cases for {combined['casrn'].nunique()} chemicals "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
