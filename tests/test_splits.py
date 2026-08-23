from __future__ import annotations

import pandas as pd

from scripts.build_demo_dataset import make_demo_dataset
from ecoood.schema import DEFAULT_SCHEMA
from ecoood.splits import (
    NAMED_CLASS_HOLDOUTS,
    balanced_group_fold_split,
    build_split,
    named_class_for_seed,
    three_way_group_split,
)


def test_chemical_random_split_assigns_each_chemical_to_one_partition() -> None:
    df = make_demo_dataset(n=240, seed=123)
    split = build_split(df, split="chemical_random", schema=DEFAULT_SCHEMA, seed=123)
    train_chemicals = set(df.loc[split.train, DEFAULT_SCHEMA.chemical_id])
    calibration_chemicals = set(df.loc[split.calib, DEFAULT_SCHEMA.chemical_id])
    test_chemicals = set(df.loc[split.test, DEFAULT_SCHEMA.chemical_id])
    assert train_chemicals.isdisjoint(test_chemicals)
    assert train_chemicals.isdisjoint(calibration_chemicals)
    assert calibration_chemicals.isdisjoint(test_chemicals)
    assert (
        len(split.train) + len(split.calib) + len(split.test)
        == len(df)
    )
    assert split.split_name == "chemical_random"
    assert split.test_is_ood.all()


def test_acyclic_scaffold_sensitivity_keeps_test_chemical_out_of_references() -> None:
    rows = []
    for chemical, smiles in (
        ("acyclic_a", "CCCC"),
        ("acyclic_b", "CCCO"),
        ("ring_a", "c1ccccc1"),
        ("ring_b", "C1CCCCC1"),
        ("ring_c", "c1ccncc1"),
    ):
        for replicate in range(8):
            rows.append(
                {
                    "chemical_id": chemical,
                    "smiles": smiles,
                    "replicate": replicate,
                }
            )
    df = pd.DataFrame(rows)
    split = build_split(
        df,
        split="scaffold_acyclic_identity",
        schema=DEFAULT_SCHEMA,
        seed=42,
    )

    test_chemicals = set(df.loc[split.test, "chemical_id"])
    reference_chemicals = set(df.loc[split.train, "chemical_id"]) | set(
        df.loc[split.calib, "chemical_id"]
    )
    assert test_chemicals.isdisjoint(reference_chemicals)
    assert split.split_name == "scaffold_acyclic_identity"


def test_chemical_class_split_holds_out_atomic_class_in_multilabel_rows() -> None:
    df = make_demo_dataset(n=240, seed=123)
    df[DEFAULT_SCHEMA.chemical_class] = "unclassified"
    held_out = named_class_for_seed(40)
    df.loc[df.index[:40], DEFAULT_SCHEMA.chemical_class] = held_out
    df.loc[df.index[40:80], DEFAULT_SCHEMA.chemical_class] = (
        f"Endocrine Disrupting Chemicals (EDCs);{held_out}"
    )
    df.loc[df.index[80:120], DEFAULT_SCHEMA.chemical_class] = (
        NAMED_CLASS_HOLDOUTS[1]
    )
    split = build_split(df, split="chemical_class", schema=DEFAULT_SCHEMA, seed=40)

    def has_held_out(value: str) -> bool:
        return held_out in {token.strip() for token in value.split(";")}

    assert df.loc[split.test, DEFAULT_SCHEMA.chemical_class].map(has_held_out).all()
    assert not df.loc[split.train, DEFAULT_SCHEMA.chemical_class].map(has_held_out).any()
    assert not df.loc[split.calib, DEFAULT_SCHEMA.chemical_class].map(has_held_out).any()
    assert pd.Series(df.loc[split.train, DEFAULT_SCHEMA.chemical_class]).eq("unclassified").any()


def test_three_way_group_split_keeps_groups_in_one_partition() -> None:
    df = pd.DataFrame(
        {
            "row": range(120),
            "reference_number": [f"ref_{index // 4:02d}" for index in range(120)],
        }
    )
    split = three_way_group_split(
        df,
        group_col="reference_number",
        split_name="reference_holdout",
        seed=123,
    )
    train_groups = set(df.loc[split.train, "reference_number"])
    calibration_groups = set(df.loc[split.calib, "reference_number"])
    test_groups = set(df.loc[split.test, "reference_number"])

    assert train_groups.isdisjoint(calibration_groups)
    assert train_groups.isdisjoint(test_groups)
    assert calibration_groups.isdisjoint(test_groups)
    assert len(split.train) + len(split.calib) + len(split.test) == len(df)


def test_balanced_group_folds_cover_each_group_once() -> None:
    df = pd.DataFrame(
        {
            "reference_number": (
                ["large_reference"] * 45
                + [
                    f"ref_{group:02d}"
                    for group in range(25)
                    for _ in range(3)
                ]
            )
        }
    )
    test_group_sets: list[set[str]] = []
    for fold_index in range(5):
        split = balanced_group_fold_split(
            df,
            group_col="reference_number",
            split_name="reference_holdout",
            fold_index=fold_index,
            n_splits=5,
            fold_seed=123,
            calib_seed=100 + fold_index,
        )
        train_groups = set(df.loc[split.train, "reference_number"])
        calibration_groups = set(df.loc[split.calib, "reference_number"])
        test_groups = set(df.loc[split.test, "reference_number"])
        assert train_groups.isdisjoint(calibration_groups)
        assert train_groups.isdisjoint(test_groups)
        assert calibration_groups.isdisjoint(test_groups)
        test_group_sets.append(test_groups)

    assert set.union(*test_group_sets) == set(df["reference_number"])
    for left in range(len(test_group_sets)):
        for right in range(left + 1, len(test_group_sets)):
            assert test_group_sets[left].isdisjoint(test_group_sets[right])
