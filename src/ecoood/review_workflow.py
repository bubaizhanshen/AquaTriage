"""Prediction and chemical-level review, using a fitted training reference."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from sklearn.neighbors import NearestNeighbors

from .ad import ApplicabilityDomainScorer, _tanimoto_novelty
from .features import EcoFeatureBuilder, RDKit_DESCRIPTOR_MAP
from .models import BootstrapEnsembleRegressor
from .molecular_identity import fixed_composition_molecule


PROTOCOL_VERSION = "aquatriage-review-1"
ENDPOINTS = {
    "fish_96h_lc50": (90.0, 102.0),
    "daphnia_48h_ec50": (42.0, 54.0),
    "algae_72_96h_ec50": (66.0, 102.0),
}
TAXONOMY = ("genus", "family", "order", "class_name", "phylum")
CONTEXT_NUMERIC = ("temperature_c", "ph", "ctx_hardness")
BIOACTIVITY = tuple(
    f"mech_{name}" for name in (
        "cytotox_hit_rate", "cytotox_lower_log10_um", "cytotox_lower_um",
        "cytotox_median_log10_um", "cytotox_median_um", "cytotox_nhit",
        "cytotox_ntested", "h295r_active_endpoint_count", "h295r_andr_max_abs",
        "h295r_bmd_min", "h295r_cort_max_abs", "h295r_cortic_max_abs",
        "h295r_critical_val", "h295r_doc_max_abs", "h295r_e1_max_abs",
        "h295r_e2_max_abs", "h295r_maxmmd_max", "h295r_mean_abs",
        "h295r_mmd_max", "h295r_ohpreg_max_abs", "h295r_ohprog_max_abs",
        "h295r_prog_max_abs", "h295r_testo_max_abs", "h295r_x11dcort_max_abs",
        "lit_ar_agonist", "lit_ar_antagonist", "lit_ar_binding",
        "lit_er_agonist", "lit_er_antagonist", "lit_er_binding",
        "toxcast_ar_auc_agonist", "toxcast_ar_auc_antagonist",
        "toxcast_er_auc_agonist", "toxcast_er_auc_antagonist",
    )
)
REQUIRED = ("chemical_id", "smiles", "species", "endpoint", "duration_h", "identity_verified")
OPTIONAL = ("chemical_name", "case_id", "medium", *TAXONOMY, *CONTEXT_NUMERIC, *BIOACTIVITY)
SCORES = ("block_knn", "similarity_ad", "ensemble_sd")
RULES = {
    "general_error": ("block_knn", "similarity_ad", "ensemble_sd"),
    "hazard_omission": ("threshold_proximity", "block_knn", "similarity_ad", "ensemble_sd"),
}
RULE_NAMES = {
    "block_knn": "Block-normalized kNN",
    "similarity_ad": "Similarity AD",
    "ensemble_sd": "Ensemble SD",
    "threshold_proximity": "Threshold proximity",
}
CASE_COLUMNS = [
    "input_row", "case_id", "chemical_id", "chemical_name", "smiles", "endpoint",
    "species", "duration_h", "molecular_weight", "prediction_log_molar",
    "prediction_log_mg_l", "prediction_mg_l", *SCORES, "missing_input_count",
    "missing_bioactivity_count", "taxonomy_filled_from_reference",
    "unseen_species", "chemical_in_training", "ids_sharing_representation",
]
NEIGHBOR_COLUMNS = [
    "case_id", "neighbor_rank", "reference_case_id", "reference_chemical_id",
    "reference_chemical_name", "reference_species", "reference_endpoint", "distance",
]
ENDPOINT_COLUMNS = [
    "chemical_id", "endpoint", "prediction_log_molar", "prediction_log_mg_l",
    *SCORES, "prediction_mg_l", "case_count",
]


def read_input_csv(
    content: bytes, *, max_bytes: int = 20 * 1024 * 1024, max_records: int = 10000,
) -> pd.DataFrame:
    """Preserve identifiers; never let CSV inference turn them into numbers."""
    if len(content) > max_bytes:
        raise ValueError(f"CSV exceeds the {max_bytes / (1024 * 1024):g} MB limit.")
    try:
        header = next(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        if len(header) != len(set(header)):
            raise ValueError("Column names must be unique.")
        frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except (StopIteration, UnicodeError, pd.errors.ParserError, csv.Error) as exc:
        raise ValueError("Use a UTF-8 CSV with one header row.") from exc
    if frame.empty:
        raise ValueError("The CSV contains no records.")
    if len(frame) > max_records:
        raise ValueError(f"Use at most {max_records:,} records per run.")
    return frame


def _present(value) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def measured_target(value, unit, molecular_weight: float) -> float:
    """Convert an uncensored positive concentration, using resolved-substance MW."""
    concentration = float(value)
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("Concentration must be positive and finite.")
    normalized = str(unit).strip().lower().replace("\u03bc", "u").replace("\u00b5", "u")
    normalized = "".join(normalized.removeprefix("ai ").split())
    molar = {"m": 1.0, "mol/l": 1.0, "mm": 1e-3, "mmol/l": 1e-3,
             "um": 1e-6, "umol/l": 1e-6, "nm": 1e-9, "nmol/l": 1e-9}
    mass = {"g/l": 1.0, "mg/l": 1e-3, "ug/l": 1e-6, "ng/l": 1e-9,
            "ppm": 1e-3, "ppb": 1e-6, "ppt": 1e-9}
    if normalized in molar:
        result = concentration * molar[normalized]
    elif normalized in mass:
        result = concentration * mass[normalized] / molecular_weight
    else:
        raise ValueError("Unsupported concentration unit.")
    if not np.isfinite(result) or result <= 0:
        raise ValueError("Invalid molar concentration.")
    return float(np.log10(result))


def prepare_inputs(frame: pd.DataFrame, *, training: bool = False):
    """Check the submitted structure and explicit identity attestation separately."""
    if frame.empty:
        raise ValueError("No input records.")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")
    missing = [name for name in REQUIRED if name not in frame]
    if training and "target_log_molar" not in frame and not {"toxicity_value", "toxicity_unit"}.issubset(frame):
        missing.append("target_log_molar (or toxicity_value and toxicity_unit)")
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    data = frame.copy().reset_index(drop=True)
    data["input_row"] = np.arange(2, len(data) + 2)
    for name in (*REQUIRED, *OPTIONAL):
        if name not in data:
            data[name] = np.nan
    reasons: list[str] = []
    canonical: list[str] = []
    weights: list[float] = []
    targets: list[float] = []
    for _, row in data.iterrows():
        issues = []
        for name in ("chemical_id", "smiles", "species", "endpoint"):
            if not _present(row[name]):
                issues.append(f"Missing {name}")
        verified = str(row["identity_verified"]).strip().lower() in {"true", "1"}
        if not verified:
            issues.append("Test-substance identity and structure not confirmed")
        endpoint = str(row["endpoint"]).strip()
        try:
            duration = float(row["duration_h"])
        except (ValueError, TypeError):
            duration = np.nan
        if endpoint not in ENDPOINTS:
            issues.append("Unsupported endpoint")
        elif not ENDPOINTS[endpoint][0] <= duration <= ENDPOINTS[endpoint][1]:
            issues.append("Duration outside endpoint window")
        structure = str(row["smiles"])
        if len(structure) > 5000:
            issues.append("SMILES exceeds the 5000-character limit")
            mol = None
        else:
            with rdBase.BlockLogs():
                mol = fixed_composition_molecule(structure)
        if mol is None:
            issues.append("No fixed-composition, carbon-containing molecular representation")
            canonical.append("")
            weights.append(np.nan)
        else:
            canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
            weights.append(float(Descriptors.MolWt(mol)))
        name = str(row["chemical_name"]).lower()
        if any(token in name for token in ("mixture", "unknown", "polymer", "organomet")):
            issues.append("Unresolved or variable-composition test substance")
        if "salt" in name and mol is not None:
            charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
            inner = (
                "inner" in name and len(Chem.GetMolFrags(mol)) == 1
                and sum(charges) == 0 and min(charges) < 0 < max(charges)
            )
            if not inner:
                issues.append("Generic salt name requires substance-specific identity review")
        medium = str(row["medium"]).strip()
        if _present(row["medium"]) and medium not in {"FW", "FW/", "AQU", "AQU/", "CUL", "CUL/"}:
            issues.append("Unsupported medium; use freshwater or culture codes")
        for column in (*CONTEXT_NUMERIC, *BIOACTIVITY):
            value = row[column]
            if _present(value):
                try:
                    number = float(value)
                    if not np.isfinite(number):
                        raise ValueError()
                    if column == "ph" and not 0 <= number <= 14:
                        raise ValueError()
                    if column == "ctx_hardness" and number < 0:
                        raise ValueError()
                except (ValueError, TypeError):
                    issues.append(f"Invalid numeric value: {column}")
        if training:
            try:
                target = (
                    float(row["target_log_molar"])
                    if "target_log_molar" in frame else
                    measured_target(row["toxicity_value"], row["toxicity_unit"], weights[-1])
                )
                if not np.isfinite(target):
                    raise ValueError()
            except (ValueError, TypeError):
                target = np.nan
                issues.append("Missing, censored, unsupported, or invalid measured concentration")
            targets.append(target)
        reasons.append("; ".join(issues))
    data["rejection_reason"] = reasons
    data["canonical_smiles"] = canonical
    data["molecular_weight"] = weights
    if training:
        data["target_log_molar"] = targets
    data["chemical_id"] = data["chemical_id"].astype(str).str.strip()
    # One identifier cannot refer to multiple structures. The reverse can occur
    # when distinct verified stereoisomers share a stereounspecified representation.
    resolved = data[data.rejection_reason.eq("")]
    inconsistent = resolved.groupby("chemical_id").canonical_smiles.nunique()
    bad_ids = set(inconsistent[inconsistent > 1].index)
    aliases = resolved.groupby("canonical_smiles").chemical_id.nunique()
    data["ids_sharing_representation"] = data.canonical_smiles.map(aliases).fillna(0).astype(int)
    conflict = data.chemical_id.isin(bad_ids)
    data.loc[conflict, "rejection_reason"] = "Inconsistent identity mapping; merge aliases or correct the structure"
    rejected = data[data.rejection_reason.ne("")][
        ["input_row", "chemical_id", "chemical_name", "smiles", "endpoint", "species", "rejection_reason"]
    ].copy()
    accepted = data[data.rejection_reason.eq("")].copy()
    for column in ("duration_h", *CONTEXT_NUMERIC, *BIOACTIVITY):
        accepted[column] = pd.to_numeric(accepted[column], errors="coerce")
    for column in ("species", "endpoint", "medium", *TAXONOMY):
        accepted[column] = accepted[column].map(lambda x: str(x).strip() if _present(x) else np.nan)
    accepted["chemical_name"] = accepted["chemical_name"].where(
        accepted["chemical_name"].map(_present), accepted["chemical_id"]
    )
    accepted["case_id"] = accepted["case_id"].where(
        accepted["case_id"].map(_present), accepted["input_row"].map(lambda x: f"row-{x}")
    ).astype(str)
    if accepted.case_id.duplicated().any():
        raise ValueError("case_id must be unique within the submitted table.")
    accepted["smiles"] = accepted["canonical_smiles"]
    for name, compute in RDKit_DESCRIPTOR_MAP.items():
        accepted[name] = [compute(Chem.MolFromSmiles(s)) for s in accepted.smiles]
    if training:
        accepted["target_log_molar"] = pd.to_numeric(accepted["target_log_molar"])
    return accepted.reset_index(drop=True), rejected.reset_index(drop=True)


def _model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "smiles", "endpoint", "species", "duration_h", "medium",
        *TAXONOMY, *CONTEXT_NUMERIC, *BIOACTIVITY, *RDKit_DESCRIPTOR_MAP,
    )
    return frame[list(columns)].copy()


@dataclass
class ReviewReference:
    builder: EcoFeatureBuilder
    ensemble: BootstrapEnsembleRegressor
    support: ApplicabilityDomainScorer
    neighbors: NearestNeighbors
    reference_cases: pd.DataFrame
    taxonomy_lookup: pd.DataFrame
    manifest: dict


def fit_reference(frame: pd.DataFrame, *, seed: int = 42, trees: int = 400) -> ReviewReference:
    data, rejected = prepare_inputs(frame, training=True)
    if len(rejected):
        examples = "; ".join(
            f"row {r.input_row}: {r.rejection_reason}" for r in rejected.head(3).itertuples()
        )
        raise ValueError(f"Resolve {len(rejected)} invalid training records before fitting. {examples}")
    if len(data) < 30 or data.chemical_id.nunique() < 10:
        raise ValueError("A training reference needs at least 30 records and 10 chemicals.")
    if data.target_log_molar.nunique() < 2:
        raise ValueError("Training targets must contain more than one value.")
    builder = EcoFeatureBuilder(include_study_year=False, recompute_rdkit_logp=True)
    features = builder.fit_transform(_model_frame(data))
    ensemble = BootstrapEnsembleRegressor(
        n_members=5, seed=seed, n_jobs=1, estimator_params={"n_estimators": trees},
    ).fit(features.full, data.target_log_molar.to_numpy())
    support = ApplicabilityDomainScorer()
    support.train_bundle = features
    support.equal_block_train = support._fit_equal_block_representation(features)
    neighbors = NearestNeighbors(n_neighbors=5, metric="euclidean", n_jobs=1)
    neighbors.fit(support.equal_block_train)
    lookup = data.groupby("species")[list(TAXONOMY)].agg(
        lambda s: s.dropna().iloc[0] if s.dropna().nunique() == 1 else np.nan
    )
    digest = hashlib.sha256(
        data[[*_model_frame(data).columns, "target_log_molar"]].to_csv(index=False).encode()
    ).hexdigest()
    import lightgbm
    import sklearn
    versions = {"rdkit": rdBase.rdkitVersion, "scikit_learn": sklearn.__version__,
                "lightgbm": lightgbm.__version__, "numpy": np.__version__}
    configuration = {"protocol": PROTOCOL_VERSION, "normalized_training_sha256": digest,
                     "seed": seed, "trees": trees, "members": 5, "software": versions}
    reference_id = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    manifest = {
        "protocol": PROTOCOL_VERSION, "reference_sha256": reference_id,
        "normalized_training_sha256": digest, "software": versions,
        "records": len(data), "chemicals": int(data.chemical_id.nunique()),
        "model": "LightGBM bootstrap ensemble", "members": 5, "trees_per_member": trees,
        "seed": seed, "rdkit": rdBase.rdkitVersion, "target_scale": "log10(mol/L)",
        "descriptor_source": "RDKit, including molecular weight and MolLogP",
        "publication_year_input": False, "k_neighbors": 5, "neighbor_unit": "training case",
        "bioactivity_variables": len(BIOACTIVITY),
        "bioactivity_observed_records": int(data[list(BIOACTIVITY)].notna().any(axis=1).sum()),
        "input_columns": list(_model_frame(data).columns),
        "feature_dimensions": len(features.feature_names),
        "block_rms_norms": support.equal_block_scale_by_name,
        "purpose": "Application fit on the supplied reference, not a held-out performance estimate",
    }
    return ReviewReference(
        builder, ensemble, support, neighbors,
        data[["case_id", "chemical_id", "chemical_name", "canonical_smiles", "species", "endpoint"]].copy(),
        lookup, manifest,
    )


def predict_cases(reference: ReviewReference, frame: pd.DataFrame):
    if reference.manifest["protocol"] != PROTOCOL_VERSION:
        raise ValueError("Reference protocol does not match this application.")
    data, rejected = prepare_inputs(frame)
    if data.empty:
        return pd.DataFrame(columns=CASE_COLUMNS), rejected, pd.DataFrame(columns=NEIGHBOR_COLUMNS)
    data["taxonomy_filled_from_reference"] = 0
    for column in TAXONOMY:
        fill = data["species"].map(reference.taxonomy_lookup[column])
        use = data[column].isna() & fill.notna()
        data.loc[use, column] = fill[use]
        data.loc[use, "taxonomy_filled_from_reference"] += 1
    optional = ["medium", *TAXONOMY, *CONTEXT_NUMERIC, *BIOACTIVITY]
    data["missing_input_count"] = data[optional].isna().sum(axis=1)
    data["missing_bioactivity_count"] = data[list(BIOACTIVITY)].isna().sum(axis=1)
    data["unseen_species"] = ~data.species.isin(reference.reference_cases.species)
    data["chemical_in_training"] = data.canonical_smiles.isin(reference.reference_cases.canonical_smiles)
    features = reference.builder.transform(_model_frame(data))
    predictions = reference.ensemble.predict(features.full)
    representation = reference.support._transform_equal_block_representation(features)
    distances, indices = reference.neighbors.kneighbors(representation)
    data["prediction_log_molar"] = predictions.mean
    data["prediction_log_mg_l"] = predictions.mean + np.log10(data.molecular_weight) + 3.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        data["prediction_mg_l"] = np.power(10.0, data.prediction_log_mg_l)
    if not (np.isfinite(data.prediction_mg_l) & data.prediction_mg_l.gt(0)).all():
        raise ValueError("Predictions exceed the numerical concentration range; check training target units.")
    data["ensemble_sd"] = predictions.std
    data["block_knn"] = distances.mean(axis=1)
    data["similarity_ad"] = _tanimoto_novelty(
        reference.support.train_bundle.fingerprint, features.fingerprint,
    )
    nearest = []
    for pos, (ids, values) in enumerate(zip(indices, distances)):
        for rank, (neighbor, distance) in enumerate(zip(ids, values), start=1):
            record = reference.reference_cases.iloc[neighbor]
            nearest.append({
                "case_id": data.iloc[pos].case_id, "neighbor_rank": rank,
                "reference_case_id": record.case_id,
                "reference_chemical_id": record.chemical_id,
                "reference_chemical_name": record.chemical_name,
                "reference_species": record.species, "reference_endpoint": record.endpoint,
                "distance": float(distance),
            })
    # Queue measurements cannot enter ranking or be mistaken for prediction outputs.
    return data[CASE_COLUMNS], rejected, pd.DataFrame(nearest, columns=NEIGHBOR_COLUMNS)


def aggregate_cases(cases: pd.DataFrame, cutoff_mg_l: float):
    if not np.isfinite(cutoff_mg_l) or cutoff_mg_l <= 0:
        raise ValueError("The concentration criterion must be positive and finite.")
    if cases.empty:
        return pd.DataFrame(columns=ENDPOINT_COLUMNS), pd.DataFrame(columns=[
            "chemical_id", "chemical_name", "prediction_mg_l", *SCORES,
            "rank", "selected_for_review", "ranking_rule", "selection_basis",
        ])
    numeric = ["prediction_log_molar", "prediction_log_mg_l", *SCORES]
    endpoints = cases.groupby(["chemical_id", "endpoint"], sort=True)[numeric].median().reset_index()
    endpoints["prediction_mg_l"] = 10.0 ** endpoints.prediction_log_mg_l
    endpoints["case_count"] = cases.groupby(["chemical_id", "endpoint"]).size().to_numpy()
    metadata = cases.groupby("chemical_id", sort=True).agg(
        chemical_name=("chemical_name", "first"), smiles=("smiles", "first"),
        case_count=("case_id", "size"),
        missing_input_count=("missing_input_count", "max"),
        ids_sharing_representation=("ids_sharing_representation", "max"),
        unseen_species=("unseen_species", "any"),
        chemical_in_training=("chemical_in_training", "any"),
    )
    chemicals = endpoints.groupby("chemical_id", sort=True).agg(
        prediction_log_mg_l=("prediction_log_mg_l", "min"),
        endpoint_count=("endpoint", "size"), **{s: (s, "max") for s in SCORES},
    ).join(metadata).reset_index()
    chemicals["prediction_mg_l"] = 10.0 ** chemicals.prediction_log_mg_l
    chemicals["predicted_high_concern"] = chemicals.prediction_log_mg_l <= np.log10(cutoff_mg_l)
    chemicals["margin_log10"] = chemicals.prediction_log_mg_l - np.log10(cutoff_mg_l)
    return endpoints, chemicals


def rank_chemicals(
    chemicals: pd.DataFrame, *, task: str, review_count: int,
    rule: str | None = None,
) -> pd.DataFrame:
    if task not in RULES:
        raise ValueError("Choose general_error or hazard_omission.")
    rule = rule or RULES[task][0]
    if rule not in RULES[task]:
        raise ValueError("The selected ranking rule does not apply to this review task.")
    if isinstance(review_count, bool) or int(review_count) != review_count or not 0 <= review_count <= len(chemicals):
        raise ValueError(f"Review count must be an integer from 0 to {len(chemicals)} eligible chemicals (requested {review_count}).")
    if chemicals.empty:
        return chemicals.copy()
    result = chemicals.copy()
    score = "margin_log10" if rule == "threshold_proximity" else rule
    required = ["chemical_id", score]
    if task == "hazard_omission":
        required.append("predicted_high_concern")
    if any(column not in result for column in required):
        raise ValueError("The chemical table does not contain the required ranking columns.")
    if result.chemical_id.isna().any() or result.chemical_id.duplicated().any():
        raise ValueError("Ranking requires one row per nonmissing chemical ID.")
    result["chemical_id"] = result["chemical_id"].astype(str)
    result[score] = pd.to_numeric(result[score], errors="coerce")
    if not np.isfinite(result[score]).all():
        raise ValueError("Ranking scores must be finite.")
    if task == "hazard_omission" and not pd.api.types.is_bool_dtype(result.predicted_high_concern):
        raise ValueError("predicted_high_concern must contain booleans.")
    columns, ascending = [score, "chemical_id"], [rule == "threshold_proximity", True]
    if task == "hazard_omission":
        columns.insert(0, "predicted_high_concern")
        ascending.insert(0, True)
    result = result.sort_values(columns, ascending=ascending, kind="stable").reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["selected_for_review"] = result["rank"] <= review_count
    result["ranking_rule"] = rule
    result["selection_basis"] = {
        "block_knn": "Greater distance from the fitted training reference",
        "similarity_ad": "Lower structural similarity to training chemicals",
        "ensemble_sd": "Greater disagreement among bootstrap predictions",
        "threshold_proximity": "Smaller predicted margin above the concentration criterion",
    }[rule]
    if task == "hazard_omission":
        result.loc[result.predicted_high_concern, "selection_basis"] = "Already predicted high concern; after the low-concern queue"
    return result


@dataclass
class ReviewRun:
    cases: pd.DataFrame
    endpoints: pd.DataFrame
    chemicals: pd.DataFrame
    rejected: pd.DataFrame
    neighbors: pd.DataFrame
    manifest: dict


def run_review(
    reference: ReviewReference, frame: pd.DataFrame, *, task: str = "general_error",
    review_fraction: float | None = 0.25, review_count: int | None = None,
    cutoff_mg_l: float = 1.0, rule: str | None = None,
) -> ReviewRun:
    cases, rejected, neighbors = predict_cases(reference, frame)
    endpoints, chemicals = aggregate_cases(cases, cutoff_mg_l)
    if review_count is None:
        if review_fraction is None or not np.isfinite(review_fraction) or not 0 <= review_fraction <= 1:
            raise ValueError("Review fraction must be between zero and one.")
        review_count = round(review_fraction * len(chemicals))
    ranked = rank_chemicals(chemicals, task=task, review_count=review_count, rule=rule)
    if not ranked.empty:
        ranked["rejected_case_count"] = ranked.chemical_id.map(rejected.chemical_id.value_counts()).fillna(0).astype(int)
    manifest = {
        "protocol": PROTOCOL_VERSION, "reference": reference.manifest,
        "input_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "task": task, "rule": rule or RULES[task][0], "cutoff_mg_l": cutoff_mg_l,
        "submitted_records": len(frame), "eligible_records": len(cases),
        "eligible_chemicals": len(chemicals), "rejected_records": len(rejected),
        "review_count": int(review_count),
        "realized_review_fraction": review_count / len(chemicals) if len(chemicals) else None,
        "quota_denominator": "all eligible chemicals",
        "fraction_rounding": "nearest integer, ties to even",
        "aggregation": "median within chemical-endpoint; minimum concentration, maximum support/SD across endpoints",
        "selection_uses_measured_queue_outcomes": False,
        "unselected_status": "not selected at this workload; not a safety or accuracy declaration",
        "excluded_input_columns": sorted(set(frame.columns) - set(REQUIRED) - set(OPTIONAL)),
    }
    return ReviewRun(cases, endpoints, ranked, rejected, neighbors, manifest)


def export_run(run: ReviewRun) -> bytes:
    """CSV cells are escaped for spreadsheet formula injection on opening."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("cases", "endpoints", "chemicals", "rejected", "neighbors"):
            table = getattr(run, name).copy()
            for column in table.select_dtypes(include=["object", "string"]):
                table[column] = table[column].map(
                    lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
                )
            archive.writestr(f"{name}.csv", table.to_csv(index=False))
        archive.writestr("run_manifest.json", json.dumps(run.manifest, indent=2, allow_nan=False))
    return output.getvalue()


def template_csv(*, training: bool = False, full: bool = False) -> bytes:
    columns = [*REQUIRED, "chemical_name", "medium", *TAXONOMY, *CONTEXT_NUMERIC]
    if full:
        columns.extend(BIOACTIVITY)
    if training:
        columns.append("target_log_molar")
    example = {
        "chemical_id": "phenol", "chemical_name": "Phenol", "smiles": "Oc1ccccc1",
        "species": "Pimephales promelas", "endpoint": "fish_96h_lc50", "duration_h": 96,
        "identity_verified": "", "medium": "FW", "genus": "Pimephales",
    }
    return pd.DataFrame([example], columns=columns).to_csv(index=False).encode()


def save_reference(reference: ReviewReference, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reference, path, compress=3)
    metadata = dict(reference.manifest)
    metadata["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def load_reference(path: Path, *, expected_sha256: str | None = None) -> ReviewReference:
    """Load only a locally generated, trusted artifact, never a web-uploaded pickle."""
    import joblib
    if expected_sha256 is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError("Reference checksum differs from its release manifest.")
    reference = joblib.load(path)
    if not isinstance(reference, ReviewReference) or reference.manifest.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("Unsupported reference artifact.")
    import lightgbm
    import sklearn
    current = {"rdkit": rdBase.rdkitVersion, "scikit_learn": sklearn.__version__,
               "lightgbm": lightgbm.__version__, "numpy": np.__version__}
    if reference.manifest.get("software") != current:
        raise ValueError("Reference software versions differ; use its recorded environment or refit the reference.")
    return reference
