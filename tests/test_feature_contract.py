from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ecoood.features import EcoFeatureBuilder
from ecoood.schema import DEFAULT_SCHEMA


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_log_molar": [-4.0, -5.0, -4.5],
            "molar_concentration": [1e-4, 1e-5, 10**-4.5],
            "toxicity_value": [10.0, 1.0, 3.2],
            "smiles": ["CCO", "CCN", "CCC"],
            "endpoint": ["fish", "daphnia", "fish"],
            "chemical_class": ["A", "B", "A"],
            "species": ["Danio rerio", "Daphnia magna", "Danio rerio"],
            "genus": ["Danio", "Daphnia", "Danio"],
            "family": ["Cyprinidae", "Daphniidae", "Cyprinidae"],
            "order": ["Cypriniformes", "Cladocera", "Cypriniformes"],
            "class_name": ["Actinopterygii", "Branchiopoda", "Actinopterygii"],
            "phylum": ["Chordata", "Arthropoda", "Chordata"],
            "trophic_group": ["fish", "invertebrate", "fish"],
            "duration_h": [96.0, 48.0, 96.0],
            "medium": ["freshwater", "freshwater", "freshwater"],
            "temperature_c": [20.0, 21.0, 19.5],
            "ph": [7.0, 7.5, 7.2],
            "study_year": [2010, 2015, 2020],
            "source": ["ecotox", "ecotox", "ecotox"],
            "ctx_hardness": [50.0, 100.0, np.nan],
            "physchem_logp": [-0.3, -0.2, 1.4],
            "mech_signal": [0.1, np.nan, 0.4],
        }
    )


def test_target_derived_columns_never_enter_predictor_matrix() -> None:
    frame = _feature_frame()
    builder = EcoFeatureBuilder(fingerprint_bits=64).fit(frame)

    assert "target_log_molar" not in builder.numeric_cols
    assert "molar_concentration" not in builder.numeric_cols
    assert DEFAULT_SCHEMA.source not in builder.categorical_cols
    assert "toxicity_value" not in builder.numeric_cols
    assert "chemical_class" not in builder.categorical_cols
    assert "trophic_group" not in builder.categorical_cols
    assert set(builder.descriptor_cols) == {
        "physchem_mol_wt",
        "physchem_logp",
        "physchem_tpsa",
        "physchem_hba",
        "physchem_hbd",
        "physchem_rot_bonds",
        "physchem_ring_count",
    }

    names = builder.tabular_feature_names()
    assert not any("target_log_molar" in name for name in names)
    assert not any("molar_concentration" in name for name in names)
    assert not any("toxicity_value" in name for name in names)


def test_bioactivity_coverage_count_is_audit_only() -> None:
    frame = _feature_frame()
    frame["mech_feature_count"] = [1.0, 0.0, 1.0]
    builder = EcoFeatureBuilder(fingerprint_bits=64).fit(frame)

    assert "mech_signal" in builder.mechanism_cols
    assert "mech_feature_count" not in builder.mechanism_cols
    assert not any("mech_feature_count" in name for name in builder.tabular_feature_names())


def test_declared_species_and_context_axes_are_materialized() -> None:
    frame = _feature_frame()
    builder = EcoFeatureBuilder(fingerprint_bits=64)
    bundle = builder.fit_transform(frame)

    assert set(builder.context_cols) == {
        "duration_h",
        "temperature_c",
        "ph",
        "ctx_hardness",
    }
    assert bundle.species.shape[1] > 0
    assert bundle.context.shape[1] > len(builder.context_cols)
    assert np.isfinite(bundle.descriptor).all()
    assert sparse.issparse(bundle.species)
    assert np.isfinite(bundle.species.data).all()
    assert np.isfinite(bundle.context).all()
    assert np.isfinite(bundle.mechanism).all()


def test_study_year_can_be_enabled_for_historical_sensitivity() -> None:
    builder = EcoFeatureBuilder(fingerprint_bits=64, include_study_year=True)
    builder.fit(_feature_frame())

    assert "study_year" in builder.context_cols


def test_rdkit_logp_replaces_sparse_linked_values() -> None:
    frame = _feature_frame()
    frame.loc[1:, "physchem_logp"] = np.nan
    builder = EcoFeatureBuilder(fingerprint_bits=64)
    bundle = builder.fit_transform(frame)

    assert np.isfinite(bundle.descriptor).all()
    assert np.unique(bundle.descriptor[:, builder.descriptor_cols.index("physchem_logp")]).size > 1
