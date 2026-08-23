from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_demo_dataset import make_demo_dataset
from ecoood.pipeline import _known_ood_labels, run_benchmark
from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import SplitIndices


def test_benchmark_smoke(tmp_path: Path) -> None:
    df = make_demo_dataset(n=240, seed=123)
    summary = run_benchmark(
        df=df,
        splits=["random", "scaffold", "chemical_random", "chemical_class", "temporal"],
        models=["random_forest"],
        output_dir=str(tmp_path / "results"),
        seed=123,
        n_members=3,
    )
    assert not summary.empty
    assert {"split", "model", "rmse", "mae", "coverage", "aurc"}.issubset(summary.columns)
    assert (tmp_path / "results" / "ood_score_summary.csv").exists()
    predictions = pd.read_csv(
        tmp_path / "results" / "random" / "random_forest" / "predictions.csv"
    )
    expected_component_ablations = {
        f"ecoood_minus_component_{component}"
        for component in (
            "d_chem_knn",
            "d_chem_mahal",
            "d_species_tax",
            "d_context",
            "context_missing_fraction",
            "d_mech",
            "bioactivity_missing_fraction",
            "u_model",
        )
    }
    assert expected_component_ablations.issubset(predictions.columns)


def test_held_out_domain_labels_follow_the_split_definition() -> None:
    test_df = pd.DataFrame({DEFAULT_SCHEMA.known_ood: [False, False]})
    split = SplitIndices(
        train=np.array([0]),
        calib=np.array([1]),
        test=np.array([2, 3]),
        split_name="scaffold",
        test_is_ood=np.array([True, True]),
    )
    assert _known_ood_labels(test_df, split, DEFAULT_SCHEMA).tolist() == [True, True]
