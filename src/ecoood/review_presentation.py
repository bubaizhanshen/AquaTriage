"""Readable views of existing review outputs; no prediction or ranking changes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .review_workflow import ENDPOINTS, ReviewRun


ENDPOINT_LABELS = {
    "fish_96h_lc50": "Fish LC50 (96 h)",
    "daphnia_48h_ec50": "Daphnid EC50 (48 h)",
    "algae_72_96h_ec50": "Algal EC50 (72-96 h)",
}
TASK_LABELS = {
    "general_error": "Check concentration accuracy",
    "hazard_omission": "Recover hazard omissions",
}
RULE_LABELS = {
    "block_knn": "Distance to training cases",
    "similarity_ad": "Structural novelty",
    "ensemble_sd": "Ensemble disagreement",
    "threshold_proximity": "Distance above concentration criterion",
}
RULE_HELP = {
    "block_knn": (
        "Block-normalized kNN: mean distance to five training cases after normalizing "
        "the five input blocks. Larger values receive higher priority. This is a "
        "training-support measure, not an error probability."
    ),
    "similarity_ad": (
        "Similarity AD: one minus the largest Morgan-fingerprint Tanimoto similarity "
        "to a training chemical. Larger values (0-1) mean less familiar structures "
        "and receive higher priority."
    ),
    "ensemble_sd": (
        "Population standard deviation of the five bootstrap predictions on the "
        "log10(mol/L) scale. Larger values receive higher priority. This is model "
        "disagreement, not a confidence interval or an error probability."
    ),
    "threshold_proximity": (
        "Log10(predicted concentration / concentration criterion). Among chemicals "
        "predicted above the criterion, smaller margins receive higher priority. "
        "Chemicals already predicted at or below the criterion enter only after "
        "that queue is exhausted."
    ),
}
SCORE_LABELS = {
    "block_knn": "Training distance",
    "similarity_ad": "Structural novelty (0-1)",
    "ensemble_sd": "Ensemble SD [log10(mol/L)]",
    "threshold_proximity": "Log10 margin above criterion",
}
CHECK_LABELS = {
    "input_row": "CSV row",
    "case_id": "Case",
    "chemical_id": "Identifier",
    "chemical_name": "Chemical",
    "smiles": "SMILES",
    "endpoint": "Endpoint",
    "species": "Species",
    "rejection_reason": "Reason",
    "missing_input_count": "Missing optional inputs",
    "missing_bioactivity_count": "Missing bioactivity inputs (of 34)",
    "taxonomy_filled_from_reference": "Taxonomic ranks filled from reference",
    "unseen_species": "Species absent from training",
    "chemical_in_training": "Structure present in training",
    "ids_sharing_representation": "Submitted IDs sharing this structure",
}


def review_queue_view(run: ReviewRun) -> pd.DataFrame:
    """Keep the active score and interpretation tied to the completed run."""
    table = run.chemicals.copy()
    rule = run.manifest["rule"]
    if table.empty:
        return table
    score = "margin_log10" if rule == "threshold_proximity" else rule
    table["review_score"] = table[score]
    table["review_measure"] = SCORE_LABELS[rule]
    table["represented_endpoints"] = table.chemical_id.map(
        run.endpoints.groupby("chemical_id").endpoint.agg(
            lambda values: "; ".join(ENDPOINT_LABELS[value] for value in values)
        )
    )
    n, quota = run.manifest["eligible_chemicals"], run.manifest["review_count"]
    allocation = f"{quota} review {'place' if quota == 1 else 'places'}"
    table["review_reason"] = [
        f"Rank {int(rank)} of {n}; {allocation}. {reason}."
        for rank, reason in zip(table["rank"], table.selection_basis)
    ]
    if run.manifest["task"] == "hazard_omission":
        table["concentration_criterion_mg_l"] = run.manifest["cutoff_mg_l"]
        table["criterion_position"] = np.where(
            table.predicted_high_concern, "At or below criterion", "Above criterion",
        )
        with np.errstate(over="ignore"):
            table["criterion_ratio"] = np.power(10.0, table.margin_log10)
        table["criterion_ratio"] = table.criterion_ratio.where(np.isfinite(table.criterion_ratio))
    return table


def endpoint_plot_table(run: ReviewRun, chemical_id: str) -> pd.DataFrame:
    """Keep unrepresented endpoints visible without inventing predictions."""
    table = pd.DataFrame({"endpoint": list(ENDPOINTS)})
    values = run.endpoints.loc[run.endpoints.chemical_id.eq(chemical_id), [
        "endpoint", "prediction_mg_l", "case_count",
    ]]
    table = table.merge(values, on="endpoint", how="left", validate="one_to_one")
    table["label"] = table.endpoint.map(ENDPOINT_LABELS)
    table["status"] = np.where(
        table.prediction_mg_l.notna(), "Predicted", "No eligible input case",
    )
    return table


def run_summary(run: ReviewRun) -> pd.DataFrame:
    manifest = run.manifest
    ref = manifest["reference"]
    rows = [
        ("Review objective", TASK_LABELS[manifest["task"]]),
        ("Ranking rule", RULE_LABELS[manifest["rule"]]),
        ("Eligible chemicals", str(manifest["eligible_chemicals"])),
        ("Selected chemicals", str(manifest["review_count"])),
        ("Review fraction", f"{manifest['realized_review_fraction']:.1%}"
         if manifest["realized_review_fraction"] is not None else "Not applicable"),
        ("Workload denominator", "All eligible chemicals"),
        ("Eligible input cases", str(manifest["eligible_records"])),
        ("Rejected input cases", str(manifest["rejected_records"])),
    ]
    if manifest["task"] == "hazard_omission":
        rows.append(("Concentration criterion", f"{manifest['cutoff_mg_l']:g} mg/L"))
    rows.extend([
        ("Prediction units", "mg/L; log10(mol/L) also included in complete results"),
        ("Chemical summary", "Lowest of the within-endpoint median predictions"),
        ("Missing endpoints", "Not assigned a prediction"),
        ("Unselected chemicals", "Not selected at this workload"),
        ("Training reference", f"{ref['records']} cases / {ref['chemicals']} chemicals"),
        ("Reference identifier", ref["reference_sha256"]),
        ("Protocol", manifest["protocol"]),
        ("Input identifier", manifest["input_sha256"]),
    ])
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def summary_text(run: ReviewRun) -> bytes:
    rows = run_summary(run)
    return (
        "AquaTriage review summary\n\n"
        + "\n".join(f"{row.Setting}: {row.Value}" for row in rows.itertuples())
        + "\n\nReview selection does not establish a prediction error or chemical safety.\n"
    ).encode("utf-8")
