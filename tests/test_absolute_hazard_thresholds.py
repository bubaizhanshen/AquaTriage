import numpy as np
import pandas as pd
import pytest

from scripts.analyze_absolute_hazard_thresholds import (
    METHODS, aggregate, arrays, curves, endpoint_match, evaluate, log_mg_l,
)


def test_mass_conversion_uses_molecular_weight():
    assert np.allclose(log_mg_l([-5, -5], [100, 200]), [0, np.log10(2)])
    with pytest.raises(ValueError):
        log_mg_l([-5], [0])


def example_frame():
    d = pd.DataFrame(dict(
        chemical_id=['a','a','b'], endpoint=['fish_96h_lc50']*3,
        duration_h=[96]*3, effect=['MOR']*3, measurement=['MORT']*3,
        physchem_mol_wt=[100]*3, truth=[-6.,-4.,-5.],
        y_pred=[-4.,-4.,-4.], lower=[-5.5]*3, upper=[-3.]*3,
        ad_equal_block_distance=[1.,2.,3.], ad_similarity=[.2,.3,.4],
        prediction_error_risk_score=[.1,.2,.3], ensemble_sd_risk=[.3,.2,.1]))
    return d


def test_boundary_inclusive_and_minimum_sensitivity():
    d = example_frame()
    median = aggregate(d, 1, 'all_median').set_index('chemical_id')
    assert median.loc['a','true_high']
    assert median.loc['b','true_high']
    assert not median.pred_high.any()
    d.loc[2,'truth'] = -5. + 1e-14
    rounded = aggregate(d, 1, 'all_median').set_index('chemical_id')
    assert rounded.loc['b','true_high']
    d.loc[1,'truth'] = -3.
    median = aggregate(d, 1, 'all_median').set_index('chemical_id')
    minimum = aggregate(d, 1, 'matched_minimum').set_index('chemical_id')
    assert not median.loc['a','true_high']
    assert minimum.loc['a','true_high']
    assert median.loc['a','block_normalized_knn'] == minimum.loc['a','block_normalized_knn']


def test_chemical_concern_is_not_endpoint_false_negative():
    d = example_frame()
    d.loc[1, 'endpoint'] = 'daphnia_48h_ec50'
    d.loc[1, 'y_pred'] = -6.
    p = aggregate(d, 1, 'all_median')
    # Chemical a is already predicted high because of its second endpoint.
    a = arrays(p)
    assert (a['true_high'] & ~a['pred_high']).sum() == 1


def test_endpoint_match_requires_effect_and_duration():
    d = example_frame()
    d.loc[0,'duration_h'] = 95
    d.loc[1,'measurement'] = 'SURV'
    assert endpoint_match(d).tolist() == [False,False,True]
    d.loc[0,['endpoint','duration_h','measurement']] = ['algae_72_96h_ec50',72,'ABND']
    d['basis_for_effect'] = ['growth rate','','']
    assert endpoint_match(d).tolist() == [True,False,True]


def test_queue_random_and_complete_curves():
    a = dict(chemical_id=np.array(list('abcdefgh')),true_high=np.array([1,0,0,0,0,0,1,1],bool),pred_high=np.array([0,0,0,0,0,0,1,1],bool))
    a.update({c:np.arange(8,0,-1,dtype=float) for c in METHODS.values()})
    result = evaluate(a)
    assert result[('random_low_queue','capture')] == 2/6
    assert result[('random_low_queue','FOR')] == 1/6
    curve = pd.DataFrame(curves(a),columns=['method','n','fraction','capture','FOR'])
    for method,g in curve.groupby('method'):
        assert len(g) == 9
        assert g.iloc[0]['capture'] == 0
        assert g.iloc[-1]['capture'] == 1
        assert np.isnan(g.iloc[-1]['FOR'])
        assert g.loc[g.n==2,'capture'].iloc[0] == result[(method,'capture')]
