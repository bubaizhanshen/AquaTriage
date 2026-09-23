"""Fixed acute-effect concentration sensitivity using locked AquaTriage outputs.

This is boundary-aligned screening, not a regulatory substance classification.
The source molecular weight reverses the original molar target conversion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.analyze_stage1_decision_baselines import (
    _aggregate_predictions, _internal_panels, _external_panels, _load_cutoff_table,
)
from scripts.analyze_stage4_paired_uncertainty import _fast_review_mask
from scripts.audit_matched_queue_protocol import METHODS

PRIMARY = ['chemical_random', 'scaffold', 'temporal', 'species']


def log_mg_l(log_molar, mw):
    mw = np.asarray(mw, dtype=float)
    if not np.all(np.isfinite(mw) & (mw > 0)):
        raise ValueError('Target-conversion molecular weights must be positive and finite.')
    return np.asarray(log_molar, dtype=float) + np.log10(mw * 1000)


def endpoint_match(frame):
    measurement = frame.measurement.fillna('').str.upper().str.rstrip('/')
    effect = frame.effect.fillna('').str.upper()
    basis = frame.get('basis_for_effect', pd.Series('', index=frame.index)).fillna('').str.lower()
    fish = frame.endpoint.eq('fish_96h_lc50') & frame.duration_h.eq(96) & effect.eq('MOR') & measurement.eq('MORT')
    daphnia = frame.endpoint.eq('daphnia_48h_ec50') & frame.duration_h.eq(48) & effect.eq('ITX') & measurement.eq('IMBL')
    algae = frame.endpoint.eq('algae_72_96h_ec50') & frame.duration_h.isin([72, 96]) & (measurement.eq('PGRT') | basis.str.contains('growth rate', regex=False))
    return fish | daphnia | algae


def arrays(panel):
    result = {c: panel[c].to_numpy() for c in ['chemical_id', 'true_high', 'pred_high', *METHODS.values()]}
    result['chemical_id'] = result['chemical_id'].astype(str)
    return result


def evaluate(a):
    n = len(a['chemical_id'])
    low = ~a['pred_high'].astype(bool)
    fn = a['true_high'].astype(bool) & low
    budget = round(.25 * n)
    result = {}
    for method, column in METHODS.items():
        mask = _fast_review_mask(a[column], a['chemical_id'], ~low, .25)
        left = low & ~mask
        result[(method, 'capture')] = (fn & mask).sum() / fn.sum() if fn.any() else np.nan
        result[(method, 'FOR')] = (fn & left).sum() / left.sum() if left.any() else np.nan
    result[('random_low_queue', 'capture')] = min(budget / low.sum(), 1) if fn.any() else np.nan
    result[('random_low_queue', 'FOR')] = fn.sum() / low.sum() if low.sum() > budget else np.nan
    return result


def curves(a):
    n = len(a['chemical_id'])
    low = ~a['pred_high'].astype(bool)
    fn = a['true_high'].astype(bool) & low
    budgets = np.arange(n + 1)
    out = []
    for method, col in METHODS.items():
        order = np.lexsort((a['chemical_id'], -a[col], ~low))
        captured = np.r_[0, np.cumsum(fn[order])]
        for k, got in zip(budgets, captured):
            remaining = max(int(low.sum()) - k, 0)
            out.append((method, k, k / n, got / fn.sum() if fn.any() else np.nan,
                        (fn.sum() - got) / remaining if remaining else np.nan))
    for k in budgets:
        p = min(k / low.sum(), 1) if low.any() else np.nan
        out.append(('random_low_queue', k, k / n, p if fn.any() else np.nan,
                    fn.sum() / low.sum() if low.sum() > k else np.nan))
    return out


def aggregate(frame, threshold, mode):
    d = frame.copy()
    if mode != 'all_median':
        d = d.loc[endpoint_match(d)].copy()
    for source, target in [('truth', 'truth_mass'), ('y_pred', 'pred_mass'), ('lower', 'lower_mass'), ('upper', 'upper_mass')]:
        d[target] = log_mg_l(d[source], d.physchem_mol_wt)
    # Aggregate concentrations separately, preserving support-score medians.
    # Snap only floating-point roundoff at the inclusive concentration boundary.
    groups = d.groupby(['chemical_id', 'endpoint'])
    for col in ['truth_mass', 'pred_mass', 'lower_mass', 'upper_mass']:
        d[col] = groups[col].transform('min' if mode == 'matched_minimum' else 'median')
        boundary = np.isclose(d[col], np.log10(threshold), rtol=0, atol=1e-12)
        d.loc[boundary, col] = np.log10(threshold)
    p = _aggregate_predictions(d, {e: np.log10(threshold) for e in d.endpoint.unique()},
        true_column='truth_mass', pred_column='pred_mass',
        interval_lower_column='lower_mass', interval_upper_column='upper_mass')
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--internal-predictions', type=Path, required=True)
    parser.add_argument('--cutoffs', type=Path, required=True)
    parser.add_argument('--external-predictions', type=Path, required=True)
    parser.add_argument('--external-panel', type=Path, default=ROOT/'data/processed/ecoood_external_expanded_v1.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-reps', type=int, default=1000)
    args = parser.parse_args()
    files = {
        'benchmark': args.benchmark,
        'internal': args.internal_predictions,
        'cutoffs': args.cutoffs,
        'external': args.external_predictions,
        'external_source': args.external_panel,
    }
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(primary_boundary_mg_l=1, sensitivity_boundaries_mg_l=[10, 100],
        aggregation_modes=['all_median', 'matched_median', 'matched_minimum'],
        interpretation='Acute-effect boundary-aligned screening, not formal GHS classification or exposure risk.',
        endpoint_match='Exact 96-h fish mortality; 48-h daphnid immobilization; 72/96-h algal PGRT or explicit growth-rate basis.',
        review='round(0.25*N); predicted-low first; random uses exactly the same queue and quota',
        bootstrap='Shared chemical multiplicities across fits/shifts; conditional on locked predictions; equal primary-shift/fit weighting',
        replicates=args.bootstrap_reps, seed=20260923,
        no_refitting=True, boundary_roundoff_tolerance_log10=1e-12,
        full_curves='Every integer review count from zero to N',
        reference='https://unece.org/info/Transport/pub/407575',
        inputs={k: dict(filename=v.name, sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in files.items()})
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    data = {k: pd.read_csv(v, low_memory=False) for k,v in files.items()}
    internal, external = data['internal'], data['external']
    source = data['benchmark'].iloc[internal.case_id.astype(int)]
    assert np.array_equal(source.chemical_id, internal.chemical_id)
    assert np.allclose(source.target_log_molar, internal.y_true)
    assert np.allclose(source.physchem_mol_wt, internal.physchem_mol_wt)
    assert not data['external_source'].case_id.duplicated().any()
    joined = external[['case_id','chemical_id','target_log_molar']].merge(data['external_source'], on='case_id', suffixes=('_pred',''), validate='many_to_one')
    assert len(joined) == len(external)
    assert np.array_equal(joined.chemical_id_pred, joined.chemical_id)
    assert np.allclose(joined.target_log_molar_pred, joined.target_log_molar)
    ext_source = data['external_source']
    direct = ext_source.toxicity_unit.eq('mg/L')
    restored = 10 ** log_mg_l(ext_source.target_log_molar, ext_source.physchem_mol_wt)
    assert np.allclose(restored[direct], ext_source.loc[direct, 'toxicity_value'])
    source_audit = ext_source[['case_id','toxicity_value','toxicity_unit','summary_value_mg_l','source_value_corrected','source_quality_status']].copy()
    source_audit['restored_mg_l'] = restored
    source_audit.to_csv(out/'source_concentration_audit.csv', index=False)
    metadata = [c for c in ['physchem_mol_wt','measurement','effect','duration_h','basis_for_effect'] if c not in external]
    external = external.merge(data['external_source'][['case_id',*metadata]], on='case_id', validate='many_to_one')
    internal = internal.assign(truth=internal.y_true, lower=internal.endpoint_interval_lower, upper=internal.endpoint_interval_upper)
    external = external.assign(truth=external.target_log_molar, lower=external.y_pred-external.interval_width/2, upper=external.y_pred+external.interval_width/2)
    audit = []
    for name, frame in [('internal', data['benchmark']), ('external', data['external_source'])]:
        for matched in [False,True]:
            d = frame.loc[endpoint_match(frame)] if matched else frame
            for endpoint,g in d.groupby('endpoint'):
                audit.append(dict(dataset=name, matched=matched, endpoint=endpoint, records=len(g), chemicals=g.chemical_id.nunique()))
    pd.DataFrame(audit).to_csv(out/'endpoint_eligibility.csv',index=False)
    control = _internal_panels(data['benchmark'], data['internal'], list(range(40,45)), _load_cutoff_table(files['cutoffs']))
    control += _external_panels(data['benchmark'], data['external'], list(range(40,45)))
    control_rows = []
    for dataset,seed,split,p in control:
        if dataset=='internal' and split not in PRIMARY: continue
        for (method,metric),value in evaluate(arrays(p)).items():
            control_rows.append(dict(dataset=dataset,seed=seed,split=split,method=method,metric=metric,value=value))
    control_df=pd.DataFrame(control_rows)
    control_df.to_csv(out/'relative_control.csv',index=False)
    print('Relative control',control_df.groupby(['dataset','method','metric']).value.mean().to_string(),flush=True)
    exact, counts, intervals, curve_rows, boot_rows = [], [], [], [], []
    for mode in protocol['aggregation_modes']:
        for threshold in [1,10,100]:
            for dataset, frame in [('internal',internal),('external',external)]:
                panels=[]
                groups=frame.groupby(['split','seed']) if dataset=='internal' else frame.groupby('seed')
                for key,g in groups:
                    split,seed=key if dataset=='internal' else ('external',key)
                    p=aggregate(g,threshold,mode)
                    a=arrays(p)
                    base=dict(mode=mode,threshold_mg_l=threshold,dataset=dataset,split=split,seed=int(seed))
                    low=~a['pred_high'].astype(bool);fn=a['true_high'].astype(bool)&low
                    counts.append(dict(**base,n=len(p),predicted_low=int(low.sum()),true_high=int(a['true_high'].sum()),false_negative=int(fn.sum()),review_n=round(.25*len(p))))
                    for (method,metric),value in evaluate(a).items():
                        exact.append(dict(**base,method=method,metric=metric,value=value))
                    for method,k,f,capture,omission in curves(a):
                        curve_rows.append(dict(**base,method=method,review_n=k,review_fraction=f,capture=capture,FOR=omission))
                    if dataset=='external' or split in PRIMARY: panels.append(a)
                ids=sorted(set(c for a in panels for c in a['chemical_id']))
                mapping={c:i for i,c in enumerate(ids)}
                indexed=[(a,np.array([mapping[c] for c in a['chemical_id']])) for a in panels]
                rng=np.random.default_rng(20260923)
                local=[]
                for rep in range(args.bootstrap_reps):
                    w=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
                    values={}
                    for a,ix in indexed:
                        take=np.repeat(np.arange(len(ix)),w[ix])
                        if not len(take):continue
                        for key,val in evaluate({c:v[take] for c,v in a.items()}).items():
                            values.setdefault(key,[]).append(val)
                    for (method,metric),vals in values.items():
                        valid=np.array(vals)[np.isfinite(vals)]
                        row=dict(mode=mode,threshold_mg_l=threshold,dataset=dataset,replicate=rep,method=method,metric=metric,value=float(valid.mean()) if len(valid) else np.nan)
                        local.append(row)
                boot_rows.extend(local)
                b=pd.DataFrame(local)
                for metric,g in b.groupby('metric'):
                    wide=g.pivot(index='replicate',columns='method',values='value')
                    for reference in ['random_low_queue','threshold']:
                        for method in wide:
                            if method==reference:continue
                            diff=(wide[method]-wide[reference]).dropna()
                            intervals.append(dict(mode=mode,threshold_mg_l=threshold,dataset=dataset,metric=metric,method=method,reference=reference,lo=diff.quantile(.025),hi=diff.quantile(.975),valid_reps=len(diff)))
                print(mode,threshold,dataset,'completed',flush=True)
    detail=pd.DataFrame(exact)
    detail.to_csv(out/'metrics_by_fit.csv',index=False)
    detail[(detail.dataset=='external')|detail.split.isin(PRIMARY)].groupby(['mode','threshold_mg_l','dataset','method','metric']).value.agg(['mean','count']).reset_index().to_csv(out/'macro_summary.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'event_counts.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(boot_rows).to_csv(out/'bootstrap.csv',index=False)
    curve_frame = pd.DataFrame(curve_rows)
    curve_frame.to_csv(out/'workload_curves.csv.gz',index=False,compression='gzip')
    area_rows = []
    keys = ['mode','threshold_mg_l','dataset','split','seed','method']
    for key, group in curve_frame.groupby(keys):
        g = group.sort_values('review_n')
        assert np.array_equal(g.review_n, np.arange(len(g)))
        if g.capture.notna().any():
            assert np.isclose(g.capture.iloc[0], 0) and np.isclose(g.capture.iloc[-1], 1)
            assert (g.capture.diff().dropna() >= -1e-10).all()
        area_rows.append(dict(zip(keys, key), capture_area=np.trapezoid(g.capture, g.review_fraction)))
    pd.DataFrame(area_rows).to_csv(out/'workload_curve_areas.csv', index=False)
    print('Outputs:',out,flush=True)


if __name__ == '__main__':
    main()
