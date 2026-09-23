from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ecoood.dsstox import normalize_casrn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "external_regulatory_prep" / "pmra_external_candidate_chemicals.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "external_regulatory_prep" / "echa_pmra_external"

ECHEM_BASE = "https://www.echemportal.org/echemportal/"
ECHEM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; EcoOOD external validation prep)",
    "Accept": "application/json",
    "Content-Type": "application/json",
}
ECHA_PARTICIPANT_ID = 821

DOSSIER_STATIC_BASE = "https://chem.echa.europa.eu/html-pages-prod/{asset_id}/"

TARGETS = {
    "fish_96h_lc50": {
        "endpoint_kind": "ShortTermToxicityToFish",
        "section_id": "id_611_Shorttermtoxicitytofish",
        "section_title": "6.1.1 Short-term toxicity to fish",
        "dose_descriptor": "LC50",
        "duration_hours": {96.0},
        "taxon_label": "freshwater fish",
    },
    "daphnia_48h_ec50": {
        "endpoint_kind": "ShortTermToxicityToAquaInv",
        "section_id": "id_613_Shorttermtoxicitytoaquaticinvertebrates",
        "section_title": "6.1.3 Short-term toxicity to aquatic invertebrates",
        "dose_descriptor": "EC50",
        "duration_hours": {48.0},
        "taxon_label": "freshwater aquatic invertebrates",
    },
    "algae_72_96h_ec50": {
        "endpoint_kind": "ToxicityToAquaticAlgae",
        "section_id": "id_615_Toxicitytoaquaticalgaeandcyanobacteria",
        "section_title": "6.1.5 Toxicity to aquatic algae and cyanobacteria",
        "dose_descriptor": "EC50",
        "duration_hours": {72.0, 96.0},
        "taxon_label": "aquatic algae and cyanobacteria",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover exact aquatic toxicity study rows for independent, multi-endpoint PMRA "
            "candidate chemicals "
            "using eChemPortal substance/property search and ECHA CHEM dossier static pages."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--request-timeout", type=int, default=60)
    return parser.parse_args()


def write_status(out_dir: Path, message: str) -> None:
    text = message.rstrip()
    (out_dir / "run_status.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    clean = clean.strip("._")
    return clean or hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def load_candidates(path: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [
        "chemical_name",
        "representative_casrn",
        "reference_codes",
        "endpoint_count",
        "min_year",
        "max_year",
    ]
    df = df[cols].drop_duplicates().reset_index(drop=True)
    df = df.rename(columns={"representative_casrn": "casrn"})
    df["casrn"] = df["casrn"].fillna("").astype(str).str.strip()
    df["casrn_normalized"] = df["casrn"].map(normalize_casrn)
    df = df[df["casrn_normalized"].ne("")].copy()
    if limit is not None:
        df = df.head(limit).copy()
    return df.reset_index(drop=True)


def cache_path(cache_dir: Path, stem: str, suffix: str) -> Path:
    return cache_dir / f"{slugify(stem)}{suffix}"


def get_json(session: requests.Session, url: str, cache_file: Path, timeout: int) -> Any:
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    response = session.get(url, timeout=timeout, headers={"Accept": "application/json"})
    response.raise_for_status()
    data = response.json()
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def post_json(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    cache_file: Path,
    timeout: int,
) -> Any:
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    response = session.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def get_text(session: requests.Session, url: str, cache_file: Path, timeout: int) -> str:
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text
    cache_file.write_text(text, encoding="utf-8")
    return text


def search_substance(session: requests.Session, casrn: str, cache_dir: Path, timeout: int) -> dict[str, Any]:
    payload = {
        "query_term": casrn,
        "paging": {"offset": 0, "limit": 10},
        "filtering": [],
        "sorting": [],
        "participants": [ECHA_PARTICIPANT_ID],
        "ghs_blocks": [],
        "new_query": True,
    }
    cache_file = cache_path(cache_dir, f"substance_search__{casrn}", ".json")
    return post_json(session, ECHEM_BASE + "api/substance-search", payload, cache_file, timeout)


def choose_best_substance_hit(result: dict[str, Any], casrn: str) -> dict[str, Any] | None:
    rows = result.get("results", [])
    if not rows:
        return None
    exact_rows = [
        row
        for row in rows
        if str(row.get("number", "")).strip() == casrn and bool(row.get("endpoint_data"))
    ]
    if not exact_rows:
        return None
    ordered = sorted(
        exact_rows,
        key=lambda row: (
            row.get("number") != casrn,
            not bool(row.get("endpoint_data")),
            row.get("level", 99),
            row.get("id", ""),
        ),
    )
    best = ordered[0]
    best = {
        "substance_id": str(best.get("id", "")),
        "substance_name": best.get("name", ""),
        "substance_number": best.get("number", ""),
        "endpoint_data": bool(best.get("endpoint_data")),
        "substance_url": best.get("url", ""),
        "participant_acronym": best.get("participant_acronym", ""),
        "level": best.get("level"),
        "remark": best.get("remark", ""),
    }
    best["rml_id"] = extract_rml_id(best.get("substance_url", ""))
    return best if best["substance_id"] else None


def extract_rml_id(url: str) -> str:
    try:
        parts = [part for part in urlparse(url).path.split("/") if part]
    except Exception:
        return ""
    return parts[0] if parts else ""


def query_property_results(
    session: requests.Session,
    substance_id: str,
    endpoint_kind: str,
    cache_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "property_blocks": [
            {
                "type": "property",
                "queryBlock": {
                    "endpointKind": endpoint_kind,
                    "queryFields": [],
                },
            }
        ],
        "paging": {"offset": 0, "limit": 300},
        "filtering": [{"field": "id", "filter": [str(substance_id)]}],
        "sorting": [],
        "participants": [ECHA_PARTICIPANT_ID],
        "new_query": True,
    }
    cache_file = cache_path(cache_dir, f"property_search__{substance_id}__{endpoint_kind}", ".json")
    return post_json(session, ECHEM_BASE + "api/property-search", payload, cache_file, timeout)


def parse_asset_id(endpoint_url: str) -> str:
    match = re.search(r"/dossier-view/([^/]+)/", endpoint_url)
    return match.group(1) if match else ""


def section_anchor_documents(index_html: str, section_id: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(index_html, "html.parser")
    section = soup.find("div", id=section_id)
    if section is None:
        return []

    docs: list[dict[str, str]] = []
    for anchor in section.select("a.das-leaf[rel='host'][href]"):
        icon = anchor.select_one("i.icon-item")
        icon_classes = icon.get("class", []) if icon else []
        icon_marker = " ".join(icon_classes)
        label = " ".join(anchor.get_text(" ", strip=True).split())
        href = anchor.get("href", "").strip()
        if not href or "_" not in href:
            continue
        docs.append(
            {
                "document_key": href,
                "label": label,
                "icon_marker": icon_marker,
                "is_study_record": (
                    "icon-ENDPOINT_STUDY_RECORD" in icon_marker
                    or (not icon_marker and bool(re.match(r"^\d+\s*\|", label)))
                ),
            }
        )
    return docs


def extract_field_pairs(document_html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(document_html, "html.parser")
    rows: list[tuple[str, str]] = []
    for field in soup.select(".das-field"):
        label_el = field.select_one(".das-field_label")
        value_el = field.select_one(".das-field_value")
        if label_el is None or value_el is None:
            continue
        label = " ".join(label_el.get_text(" ", strip=True).split())
        value = " ".join(value_el.get_text(" ", strip=True).split())
        if label:
            rows.append((label, value))
    return rows


def first_nonempty(fields: list[tuple[str, str]], labels: list[str]) -> str:
    wanted = {label.lower() for label in labels}
    for label, value in fields:
        if label.lower() in wanted and value and value not in {"[Empty]", "[Not publishable]"}:
            return value
    return ""


def linked_field_documents(document_html: str, field_label: str) -> list[str]:
    """Keep reference/material links: their content is not inline in study pages."""
    soup = BeautifulSoup(document_html, "html.parser")
    keys = set()
    for field in soup.select('.das-field'):
        label = field.select_one('.das-field_label')
        if label is None or label.get_text(' ', strip=True).casefold() != field_label.casefold():
            continue
        for link in field.select('a.das-field_reference-link[href]'):
            key = link['href'].strip()
            if re.fullmatch(r'[A-Za-z0-9_-]+', key):
                keys.add(key)
    return sorted(keys)


def parse_duration_hours(text: str) -> float | None:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(h|hours?|d|days?)\s*", text.lower())
    if not match:
        return None
    return float(match.group(1)) * (24 if match.group(2).startswith('d') else 1)


def extract_result_blocks(fields: list[tuple[str, str]]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    result_labels = {
        "Duration",
        "Dose descriptor",
        "Effect conc.",
        "95% CI",
        "Nominal / measured",
        "Conc. based on",
        "Basis for effect",
        "Remarks on result",
        "Key result",
    }
    stop_labels = {
        "Details on results",
        "Results with reference substance (positive control)",
        "Reported statistics and error estimates",
    }
    for label, value in fields:
        if label == "Duration":
            if current and current.get("Duration") and current.get("Effect conc."):
                blocks.append(current)
            current = {"Duration": value}
            continue
        if current is not None and label in result_labels:
            current[label] = value
            continue
        if current is not None and label in stop_labels:
            if current.get("Duration") and current.get("Effect conc."):
                blocks.append(current)
            current = None
    if current and current.get("Duration") and current.get("Effect conc."):
        blocks.append(current)
    return blocks


def normalize_descriptor(text: str) -> str:
    return text.replace("other:", "").strip().upper()


def parse_effect_value(text: str) -> tuple[float | None, str, bool]:
    clean = text.replace("Âµ", "µ").replace("μ", "µ").strip()
    is_censored = bool(re.search(r"(?:<=|>=|<|>|≤|≥)", clean))
    # Do not turn a reported range or approximation into an exact endpoint.
    is_censored = is_censored or bool(re.search(
        r"(?:\b(?:ca\.?|approx(?:imately)?\.?)\s|[~≈]|"
        r"\d\s*(?:-|–|—|\bto\b)\s*\d)", clean, re.IGNORECASE
    ))
    # A decimal comma is ambiguous; only unambiguous thousands groups are removed.
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", clean):
        if ',' in token and not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
            is_censored = True
    numeric_text = clean.replace(",", "")
    unit_match = re.search(
        r"([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)\s*"
        r"(mol/(?:L|l)|mmol/(?:L|l)|µmol/(?:L|l)|umol/(?:L|l)|"
        r"nmol/(?:L|l)|pmol/(?:L|l)|mg/(?:mL|ml)|µg/(?:mL|ml)|"
        r"ug/(?:mL|ml)|ng/(?:mL|ml)|g/(?:L|l)|mg/(?:L|l)|"
        r"µg/(?:L|l)|ug/(?:L|l)|ng/(?:L|l)|pg/(?:L|l)|"
        r"mM|µM|uM|nM|pM|M|ppm|ppb|ppt)\b",
        numeric_text,
        flags=re.IGNORECASE,
    )
    if unit_match:
        unit = (
            unit_match.group(2)
            .replace("ug", "µg")
            .replace("/l", "/L")
            .replace("/ml", "/mL")
        )
        value = None if is_censored else float(unit_match.group(1))
        if value is not None and value <= 0:
            value = None
        return value, unit, is_censored

    value_match = re.search(
        r"([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)", numeric_text
    )
    if not value_match:
        return None, "", is_censored
    value = None if is_censored else float(value_match.group(1))
    if value is not None and value <= 0:
        value = None
    return value, "", is_censored


def document_to_exact_rows(
    *,
    chemical_name: str,
    casrn: str,
    rml_id: str,
    asset_id: str,
    target_endpoint: str,
    section_title: str,
    document_key: str,
    document_label: str,
    document_html: str,
) -> list[dict[str, Any]]:
    target = TARGETS[target_endpoint]
    fields = extract_field_pairs(document_html)
    species = first_nonempty(
        fields,
        ["Test organisms (species)", "Test organisms (species) / cell line"],
    )
    total_duration = first_nonempty(fields, ["Total exposure duration"])
    nominal_measured = first_nonempty(fields, ["Nominal and measured concentrations"])
    test_type = first_nonempty(fields, ["Test type"])
    details_conditions = first_nonempty(fields, ["Details on test conditions"])
    reference_keys = ";".join(linked_field_documents(document_html, 'Reference'))
    material_keys = ";".join(linked_field_documents(document_html, 'Test material information'))

    rows: list[dict[str, Any]] = []
    for block in extract_result_blocks(fields):
        descriptor = normalize_descriptor(block.get("Dose descriptor", ""))
        duration_h = parse_duration_hours(block.get("Duration", ""))
        if descriptor != target["dose_descriptor"]:
            continue
        if duration_h is None or duration_h not in target["duration_hours"]:
            continue
        effect_text = block.get("Effect conc.", "")
        effect_value, effect_unit, is_censored = parse_effect_value(effect_text)
        if effect_value is None and not is_censored:
            continue
        rows.append(
            {
                "chemical_name": chemical_name,
                "casrn": casrn,
                "rml_id": rml_id,
                "asset_id": asset_id,
                "target_endpoint": target_endpoint,
                "section_title": section_title,
                "document_key": document_key,
                "document_label": document_label,
                "regulatory_species": species,
                "regulatory_taxon": target["taxon_label"],
                "total_exposure_duration": total_duration,
                "duration_h_used": duration_h,
                "dose_descriptor": descriptor,
                "source_value": effect_value,
                "source_unit": effect_unit,
                "source_effect_text": effect_text,
                "source_is_censored": is_censored,
                "nominal_or_measured": block.get("Nominal / measured", ""),
                "conc_based_on": block.get("Conc. based on", ""),
                "basis_for_effect": block.get("Basis for effect", ""),
                "remarks_on_result": block.get("Remarks on result", ""),
                "study_type": test_type,
                "type_of_information": first_nonempty(fields, ["Type of information", "Study result type"]),
                "study_reliability": first_nonempty(fields, ["Reliability"]),
                "study_adequacy": first_nonempty(fields, ["Adequacy of study"]),
                "test_material_details": first_nonempty(fields, ["Details on test material", "Specific details on test material used for the study"]),
                "test_material_identity": first_nonempty(fields, ["Test material identity"]),
                "reference_author": first_nonempty(fields, ["Author"]),
                "reference_year": first_nonempty(fields, ["Year"]),
                "reference_title": first_nonempty(fields, ["Title"]),
                "reference_report_number": first_nonempty(fields, ["Report no.", "Report number"]),
                "test_medium": first_nonempty(fields, ["Water media type"]),
                "reference_document_keys": reference_keys,
                "test_material_document_keys": material_keys,
                "details_on_test_conditions": details_conditions,
                "nominal_measured_context": nominal_measured,
                "document_url": DOSSIER_STATIC_BASE.format(asset_id=asset_id) + f"documents/{document_key}.html",
            }
        )
    return rows


def summarize_exact_rows(exact_rows: pd.DataFrame) -> list[str]:
    if exact_rows.empty:
        return ["Target-endpoint result rows extracted: 0"]
    summary = [
        f"Target-endpoint result rows extracted (including flagged bounds/ranges): {len(exact_rows)}",
        "Extraction is not eligibility approval; experimental origin, material, identity and endpoint still require curation.",
        f"Unique chemicals: {exact_rows['casrn'].nunique()}",
        f"Unique dossiers: {exact_rows['asset_id'].nunique()}",
    ]
    for endpoint, frame in exact_rows.groupby("target_endpoint"):
        summary.append(
            f"{endpoint}: {len(frame)} rows across {frame['casrn'].nunique()} chemicals and {frame['asset_id'].nunique()} dossiers"
        )
    return summary


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(out_dir / "cache")
    substance_cache = ensure_dir(cache_dir / "substance_search")
    property_cache = ensure_dir(cache_dir / "property_search")
    dossier_cache = ensure_dir(cache_dir / "dossier_index")
    document_cache = ensure_dir(cache_dir / "documents")

    write_status(out_dir, "Loading independent multi-endpoint PMRA candidates.")
    candidates = load_candidates(args.input, args.limit)

    session = requests.Session()
    session.headers.update(ECHEM_HEADERS)

    matched_rows: list[dict[str, Any]] = []
    property_rows: list[dict[str, Any]] = []
    doc_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []

    total = len(candidates)
    for idx, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        chemical_name = str(candidate["chemical_name"]).strip()
        casrn = str(candidate["casrn"]).strip()
        write_status(out_dir, f"[{idx}/{total}] Searching ECHA REACH for {chemical_name} ({casrn}).")

        try:
            substance_result = search_substance(session, casrn, substance_cache, args.request_timeout)
            best_hit = choose_best_substance_hit(substance_result, casrn)
        except Exception as exc:
            matched_rows.append(
                {
                    **candidate.to_dict(),
                    "matched_to_echa": False,
                    "match_error": str(exc),
                }
            )
            time.sleep(args.sleep)
            continue

        if best_hit is None:
            matched_rows.append(
                {
                    **candidate.to_dict(),
                    "matched_to_echa": False,
                    "substance_search_result_count": len(substance_result.get("results", [])),
                }
            )
            time.sleep(args.sleep)
            continue

        matched_rows.append(
            {
                **candidate.to_dict(),
                **best_hit,
                "matched_to_echa": True,
                "substance_search_result_count": len(substance_result.get("results", [])),
            }
        )

        seen_docs: set[tuple[str, str, str]] = set()
        for target_endpoint, target in TARGETS.items():
            try:
                prop_result = query_property_results(
                    session,
                    best_hit["substance_id"],
                    target["endpoint_kind"],
                    property_cache,
                    args.request_timeout,
                )
            except Exception as exc:
                property_rows.append(
                    {
                        "chemical_name": chemical_name,
                        "casrn": casrn,
                        "substance_id": best_hit["substance_id"],
                        "target_endpoint": target_endpoint,
                        "endpoint_kind": target["endpoint_kind"],
                        "property_query_error": str(exc),
                    }
                )
                continue

            endpoint_results = prop_result.get("results", [])
            page_info = prop_result.get("page_info", {}) or {}
            assets = sorted(
                {
                    parse_asset_id(row.get("endpoint_url", ""))
                    for row in endpoint_results
                    if parse_asset_id(row.get("endpoint_url", ""))
                }
            )
            property_rows.append(
                {
                    "chemical_name": chemical_name,
                    "casrn": casrn,
                    "substance_id": best_hit["substance_id"],
                    "rml_id": best_hit["rml_id"],
                    "target_endpoint": target_endpoint,
                    "endpoint_kind": target["endpoint_kind"],
                    "property_result_count": len(endpoint_results),
                    "property_total_elements": page_info.get("total_elements"),
                    "property_assets": "; ".join(assets),
                    "first_endpoint_url": endpoint_results[0].get("endpoint_url", "") if endpoint_results else "",
                }
            )

            for asset_id in assets:
                try:
                    index_html = get_text(
                        session,
                        DOSSIER_STATIC_BASE.format(asset_id=asset_id) + "index.html",
                        cache_path(dossier_cache, asset_id, ".html"),
                        args.request_timeout,
                    )
                except Exception:
                    continue

                docs = section_anchor_documents(index_html, target["section_id"])
                for doc in docs:
                    doc_key = doc["document_key"]
                    key_tuple = (target_endpoint, asset_id, doc_key)
                    if key_tuple in seen_docs:
                        continue
                    seen_docs.add(key_tuple)
                    doc_rows.append(
                        {
                            "chemical_name": chemical_name,
                            "casrn": casrn,
                            "rml_id": best_hit["rml_id"],
                            "asset_id": asset_id,
                            "target_endpoint": target_endpoint,
                            "section_title": target["section_title"],
                            **doc,
                        }
                    )
                    if not doc["is_study_record"]:
                        continue

                    try:
                        doc_html = get_text(
                            session,
                            DOSSIER_STATIC_BASE.format(asset_id=asset_id) + f"documents/{doc_key}.html",
                            cache_path(document_cache, f"{asset_id}__{doc_key}", ".html"),
                            args.request_timeout,
                        )
                    except Exception:
                        continue

                    exact_rows.extend(
                        document_to_exact_rows(
                            chemical_name=chemical_name,
                            casrn=casrn,
                            rml_id=best_hit["rml_id"],
                            asset_id=asset_id,
                            target_endpoint=target_endpoint,
                            section_title=target["section_title"],
                            document_key=doc_key,
                            document_label=doc["label"],
                            document_html=doc_html,
                        )
                    )
            time.sleep(args.sleep)

    matched_df = pd.DataFrame(matched_rows)
    property_df = pd.DataFrame(property_rows)
    doc_df = pd.DataFrame(doc_rows)
    exact_df = pd.DataFrame(exact_rows)

    if not exact_df.empty:
        exact_df = exact_df.drop_duplicates(
            subset=[
                "casrn",
                "asset_id",
                "document_key",
                "target_endpoint",
                "duration_h_used",
                "dose_descriptor",
                "source_value",
                "basis_for_effect",
                "regulatory_species",
            ]
        ).reset_index(drop=True)

    matched_df.to_csv(out_dir / "echa_pmra_substance_matches.csv", index=False)
    property_df.to_csv(out_dir / "echa_pmra_property_probe.csv", index=False)
    doc_df.to_csv(out_dir / "echa_pmra_toc_documents.csv", index=False)
    exact_df.to_csv(out_dir / "echa_pmra_exact_rows.csv", index=False)

    summary_lines = [
        f"Candidates processed: {len(candidates)}",
        f"ECHA matches: {int(matched_df.get('matched_to_echa', pd.Series(dtype=bool)).fillna(False).sum()) if not matched_df.empty else 0}",
        f"Property probe rows: {len(property_df)}",
        f"TOC document rows: {len(doc_df)}",
        *summarize_exact_rows(exact_df),
    ]
    (out_dir / "echa_pmra_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_status(out_dir, "Completed ECHA PMRA external-row extraction.")


if __name__ == "__main__":
    main()
