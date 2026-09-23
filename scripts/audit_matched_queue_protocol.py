"""Matched-queue controls and paired, shared-chemical bootstrap of macro means.

Reuses locked predictions; does not select candidates or refit any model.
The bootstrap conditions on fitted models and observed shifts. The same sampled
chemical multiplicity is used across every fit/shift containing that chemical.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_stage1_decision_baselines import (
    _internal_panels, _external_panels, _load_cutoff_table,
)
from scripts.analyze_stage4_paired_uncertainty import _fast_review_mask

METHODS = {
    "threshold": "prediction_distance",
    "interval": "interval_cross_score",
    "block_knn": "block_normalized_knn",
    "risk": "prediction_error_risk",
    "ensemble": "ensemble_sd_risk",
    "similarity": "similarity_ad",
}


def evaluate(a):
    n = len(a["chemical_id"])
    if not n:
        return {}
    low = ~a["pred_high"].astype(bool)
    fn = a["true_high"].astype(bool) & low
    k = round(.25 * n)
    first = _fast_review_mask(a["prediction_distance"], a["chemical_id"], ~low, .25)
    residual = low & ~first
    extra = min(k, int(residual.sum()))
    result = {}
    def metrics(mask):
        left = low & ~mask
        return (float((fn & mask).sum() / fn.sum()) if fn.any() else np.nan,
                float((fn & left).sum() / left.sum()) if left.any() else np.nan)
    for name, column in METHODS.items():
        mask = _fast_review_mask(a[column], a["chemical_id"], ~low, .25)
        assert int(mask.sum()) == min(k, n)
        if int(low.sum()) >= k:
            assert not np.any(mask & ~low)
        capture, omission = metrics(mask)
        result[("single", name, "capture")] = capture
        result[("single", name, "FOR")] = omission
        ix = np.flatnonzero(residual)
        second = np.zeros(n, dtype=bool)
        if len(ix):
            sub = _fast_review_mask(a[column][ix], a["chemical_id"][ix],
                                    np.zeros(len(ix), dtype=bool), extra / len(ix), low_only=False)
            second[ix[sub]] = True
        result[("second", name, "residual_capture")] = (
            float((fn & second).sum() / (fn & residual).sum())
            if (fn & residual).any() else np.nan)
        capture, omission = metrics(first | second)
        result[("combined", name, "capture")] = capture
        result[("combined", name, "FOR")] = omission
    p = min(k / int(low.sum()), 1) if low.any() else np.nan
    result[("single", "random_low_queue", "capture")] = p if fn.any() else np.nan
    result[("single", "random_low_queue", "FOR")] = (
        float(fn.sum() / low.sum()) if low.sum() > k else np.nan)
    result[("second", "random_low_queue", "residual_capture")] = (
        extra / int(residual.sum()) if (fn & residual).any() else np.nan)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--internal-data", type=Path, required=True)
    parser.add_argument("--internal-predictions", type=Path, required=True)
    parser.add_argument("--internal-cutoffs", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.internal_data, low_memory=False)
    pred = pd.read_csv(args.internal_predictions, low_memory=False)
    ext = pd.read_csv(args.external_predictions, low_memory=False)
    panels = _internal_panels(data, pred, list(range(40,45)), _load_cutoff_table(args.internal_cutoffs))
    panels = [p for p in panels if p[2] in {"chemical_random", "scaffold", "temporal", "species"}]
    panels += _external_panels(data, ext, list(range(40,45)))
    exact, estimates = [], []
    rng = np.random.default_rng(20260920)
    for dataset in ("internal", "external"):
        subset = [p for p in panels if p[0] == dataset]
        ids = sorted(set(str(c) for _,_,_,p in subset for c in p.chemical_id))
        index = {c:i for i,c in enumerate(ids)}
        arrays = []
        for _, seed, split, p in subset:
            a = {c:p[c].to_numpy() for c in ["chemical_id","true_high","pred_high",*METHODS.values()]}
            a["chemical_id"] = a["chemical_id"].astype(str)
            arrays.append((a, np.array([index[c] for c in a["chemical_id"]])))
            for (stage, method, metric), value in evaluate(a).items():
                exact.append(dict(dataset=dataset, seed=seed, split=split, stage=stage, method=method, metric=metric, value=value))
        for rep in range(args.reps):
            weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)))
            values = {}
            for a, ix in arrays:
                sampled = np.repeat(np.arange(len(ix)), weights[ix])
                for key, val in evaluate({c:v[sampled] for c,v in a.items()}).items():
                    values.setdefault(key, []).append(val)
            for (stage, method, metric), vals in values.items():
                estimates.append(dict(dataset=dataset, replicate=rep, stage=stage, method=method, metric=metric, value=float(np.nanmean(vals))))
    exact = pd.DataFrame(exact)
    boot = pd.DataFrame(estimates)
    exact.to_csv(args.output_dir / "exact_panels.csv", index=False)
    boot.to_csv(args.output_dir / "macro_bootstrap.csv", index=False)
    exact.groupby(["dataset","stage","method","metric"], as_index=False).value.mean().to_csv(args.output_dir / "macro_estimates.csv", index=False)
    differences = []
    for (dataset,stage,metric), group in boot.groupby(["dataset","stage","metric"]):
        wide = group.pivot(index="replicate", columns="method", values="value")
        for reference in ["random_low_queue", "threshold"]:
            if reference not in wide:
                continue
            for method in wide:
                if method == reference:
                    continue
                d = (wide[method]-wide[reference]).dropna()
                differences.append(dict(dataset=dataset,stage=stage,metric=metric,method=method,reference=reference,
                    bootstrap_mean=float(d.mean()),lo=float(d.quantile(.025)),hi=float(d.quantile(.975)),valid_reps=len(d)))
    pd.DataFrame(differences).to_csv(args.output_dir / "paired_macro_intervals.csv", index=False)
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "seed":20260920,"replicates":args.reps,"benchmark_cases":len(data),
        "bootstrap":"shared chemical multiplicity across fits and shifts; average panels within each replicate before percentiles",
        "scope":"conditional on observed shifts, fitted models and fixed calibration cutoffs",
        "budget":"round(0.25*N) first stage plus the same count in remaining predicted-low queue, capped by queue size",
        "selection":"not evaluated as part of this two-stage experiment"},indent=2))
    print(exact.groupby(["dataset","stage","method","metric"]).value.mean().to_string())


if __name__ == "__main__":
    main()
