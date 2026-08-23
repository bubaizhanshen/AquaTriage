from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

from ecoood.dsstox import normalize_casrn


ECHEMPORTAL_API = "https://www.echemportal.org/echemportal/api/property-search"
ECHA_PARTICIPANT_ID = 821
ENDPOINT_KINDS = {
    "fish_96h_lc50": "ShortTermToxicityToFish",
    "daphnia_48h_ec50": "ShortTermToxicityToAquaInv",
    "algae_72_96h_ec50": "ToxicityToAquaticAlgae",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic ECHA extension set using chemical identity "
            "and acute-endpoint availability only."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--existing-external",
        type=Path,
        default=None,
        help="Optional earlier external set to exclude by CASRN.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=150)
    parser.add_argument("--page-size", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=90)
    return parser.parse_args()


def endpoint_page(
    session: requests.Session,
    endpoint_kind: str,
    *,
    offset: int,
    limit: int,
    timeout: int,
) -> dict[str, object]:
    payload = {
        "property_blocks": [
            {
                "type": "property",
                "queryBlock": {"endpointKind": endpoint_kind, "queryFields": []},
            }
        ],
        "paging": {"offset": offset, "limit": limit},
        "filtering": [],
        "sorting": [],
        "participants": [ECHA_PARTICIPANT_ID],
        "new_query": offset == 0,
    }
    response = session.post(ECHEMPORTAL_API, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def collect_endpoint_availability(
    session: requests.Session,
    endpoint_name: str,
    endpoint_kind: str,
    *,
    cache_dir: Path,
    page_size: int,
    timeout: int,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    offset = 0
    while True:
        cache_file = cache_dir / f"{endpoint_name}_{offset:05d}.json"
        if cache_file.exists():
            response = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            response = endpoint_page(
                session,
                endpoint_kind,
                offset=offset,
                limit=page_size,
                timeout=timeout,
            )
            cache_file.write_text(
                json.dumps(response, ensure_ascii=False), encoding="utf-8"
            )

        page_rows = response.get("results", [])
        if not isinstance(page_rows, list) or not page_rows:
            break
        for record in page_rows:
            if str(record.get("number_type", "")) != "CAS Number":
                continue
            casrn = str(record.get("number", "")).strip()
            normalized = normalize_casrn(casrn)
            if not normalized:
                continue
            rows.append(
                {
                    "endpoint": endpoint_name,
                    "casrn": casrn,
                    "casrn_normalized": normalized,
                    "chemical_name": str(record.get("name", "")).strip(),
                }
            )
        if len(page_rows) < page_size:
            break
        offset += page_size
    return pd.DataFrame(rows).drop_duplicates()


def casrn_set(path: Path | None) -> set[str]:
    if path is None:
        return set()
    frame = pd.read_csv(path, usecols=["casrn"])
    return {
        normalized
        for value in frame["casrn"]
        if (normalized := normalize_casrn(value))
    }


def main() -> None:
    args = parse_args()
    if args.candidate_limit < 1:
        raise ValueError("--candidate-limit must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "property_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    endpoint_frames = {
        name: collect_endpoint_availability(
            session,
            name,
            kind,
            cache_dir=cache_dir,
            page_size=args.page_size,
            timeout=args.request_timeout,
        )
        for name, kind in ENDPOINT_KINDS.items()
    }
    availability = pd.concat(endpoint_frames.values(), ignore_index=True)
    availability.to_csv(
        args.output_dir / "echa_endpoint_availability.csv", index=False
    )

    endpoint_counts = availability.groupby("casrn_normalized")["endpoint"].nunique()
    eligible_ids = set(endpoint_counts[endpoint_counts.eq(len(ENDPOINT_KINDS))].index)
    universe = (
        availability[availability["casrn_normalized"].isin(eligible_ids)]
        .sort_values(["casrn_normalized", "chemical_name"])
        .drop_duplicates("casrn_normalized")
        [["chemical_name", "casrn", "casrn_normalized"]]
        .copy()
    )

    benchmark_ids = casrn_set(args.benchmark)
    prior_external_ids = casrn_set(args.existing_external)
    universe["benchmark_overlap"] = universe["casrn_normalized"].isin(benchmark_ids)
    universe["prior_external_overlap"] = universe["casrn_normalized"].isin(
        prior_external_ids
    )
    universe["selection_key"] = universe["casrn_normalized"].map(
        lambda value: hashlib.sha256(value.encode("ascii")).hexdigest()
    )
    universe = universe.sort_values(["selection_key", "casrn_normalized"])
    universe.to_csv(args.output_dir / "echa_three_endpoint_universe.csv", index=False)

    selected = universe.loc[
        ~universe["benchmark_overlap"] & ~universe["prior_external_overlap"]
    ].head(args.candidate_limit)
    candidate_table = selected.assign(
        representative_casrn=selected["casrn"],
        reference_codes="ECHA endpoint-availability sample",
        endpoint_count=len(ENDPOINT_KINDS),
        min_year=pd.NA,
        max_year=pd.NA,
    )[
        [
            "chemical_name",
            "representative_casrn",
            "reference_codes",
            "endpoint_count",
            "min_year",
            "max_year",
        ]
    ]
    candidate_table.to_csv(
        args.output_dir / "echa_extension_candidates.csv", index=False
    )

    summary = {
        "endpoint_rows": {
            name: int(len(frame)) for name, frame in endpoint_frames.items()
        },
        "all_three_endpoint_chemicals": int(len(universe)),
        "benchmark_disjoint_chemicals": int((~universe["benchmark_overlap"]).sum()),
        "selected_candidates": int(len(candidate_table)),
        "selection_inputs": ["chemical identity", "endpoint availability"],
        "selection_uses_toxicity_or_predictions": False,
    }
    (args.output_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
