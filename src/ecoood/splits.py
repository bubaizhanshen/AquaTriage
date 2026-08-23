from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import make_scaffold
from .schema import DEFAULT_SCHEMA, EcoOODSchema


NAMED_CLASS_HOLDOUTS = (
    "Per- and Polyfluoroalkyl Substances (PFAS)",
    "Conazoles",
    "Neonicotinoids",
    "Pharmaceutical Personal Care Products (PPCPs)",
    "Strobins",
)
NAMED_CLASS_SEED_OFFSET = 40


@dataclass
class SplitIndices:
    train: np.ndarray
    calib: np.ndarray
    test: np.ndarray
    split_name: str
    test_is_ood: np.ndarray


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _partition(indices: np.ndarray, rng: np.random.Generator, calib_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    n_calib = max(1, int(round(len(shuffled) * calib_fraction)))
    calib = shuffled[:n_calib]
    train = shuffled[n_calib:]
    return train, calib


def random_split(
    df: pd.DataFrame,
    seed: int = 42,
    test_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    rng = _rng(seed)
    indices = df.index.to_numpy()
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    test = shuffled[:n_test]
    remaining = shuffled[n_test:]
    train, calib = _partition(remaining, rng, calib_fraction)
    mask = np.zeros(len(test), dtype=bool)
    return SplitIndices(train=train, calib=calib, test=test, split_name="random", test_is_ood=mask)


def scaffold_split(
    df: pd.DataFrame,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    working = df.copy()
    working["_scaffold"] = working[schema.smiles].map(make_scaffold)
    return group_holdout_split(
        working,
        group_col="_scaffold",
        split_name="scaffold",
        seed=seed,
        holdout_fraction=holdout_fraction,
        calib_fraction=calib_fraction,
    )


def scaffold_acyclic_identity_split(
    df: pd.DataFrame,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    """Scaffold sensitivity that keeps each acyclic chemical as one group.

    Standard Bemis-Murcko processing returns an empty scaffold for every
    acyclic molecule. The primary scaffold split retains that conventional
    definition. This sensitivity prevents the chemically heterogeneous empty
    scaffold from becoming one oversized test group while keeping all records
    for each acyclic chemical together.
    """
    working = df.copy()
    scaffolds = working[schema.smiles].map(make_scaffold).astype(str)
    chemical_ids = working[schema.chemical_id].fillna("__missing_chemical__").astype(str)
    working["_scaffold_acyclic_identity"] = np.where(
        scaffolds.eq("acyclic"),
        "acyclic::" + chemical_ids,
        scaffolds,
    )
    return group_holdout_split(
        working,
        group_col="_scaffold_acyclic_identity",
        split_name="scaffold_acyclic_identity",
        seed=seed,
        holdout_fraction=holdout_fraction,
        calib_fraction=calib_fraction,
    )


def chemical_random_split(
    df: pd.DataFrame,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    """Assign each chemical wholly to train, calibration, or test."""
    return three_way_group_split(
        df,
        group_col=schema.chemical_id,
        split_name="chemical_random",
        seed=seed,
        holdout_fraction=holdout_fraction,
        calib_fraction=calib_fraction,
    )


def three_way_group_split(
    df: pd.DataFrame,
    group_col: str,
    split_name: str,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    """Assign each group wholly to train, calibration, or test."""
    if group_col not in df.columns:
        raise KeyError(f"Missing grouping column '{group_col}'.")
    rng = _rng(seed)
    labels = df[group_col].fillna("__missing_group__").astype(str)
    group_sizes = labels.groupby(labels, dropna=False).size()
    groups = group_sizes.index.to_list()
    if len(groups) < 3:
        raise ValueError("Three-way grouped splitting requires at least three groups.")
    rng.shuffle(groups)

    def take_groups(
        candidates: list[str],
        target_rows: int,
        *,
        min_remaining_groups: int,
    ) -> tuple[list[str], list[str]]:
        selected: list[str] = []
        selected_rows = 0
        selectable = max(0, len(candidates) - min_remaining_groups)
        for group in candidates[:selectable]:
            selected.append(group)
            selected_rows += int(group_sizes.loc[group])
            if selected_rows >= target_rows:
                break
        if not selected:
            raise ValueError("Grouped split could not allocate a nonempty partition.")
        return selected, candidates[len(selected) :]

    n_test_target = max(1, int(round(len(df) * holdout_fraction)))
    test_groups, remaining_groups = take_groups(
        groups,
        n_test_target,
        min_remaining_groups=2,
    )
    n_remaining = int(group_sizes.loc[remaining_groups].sum())
    n_calib_target = max(1, int(round(n_remaining * calib_fraction)))
    calib_groups, train_groups = take_groups(
        remaining_groups,
        n_calib_target,
        min_remaining_groups=1,
    )

    test = df.index[labels.isin(test_groups)].to_numpy()
    calib = df.index[labels.isin(calib_groups)].to_numpy()
    train = df.index[labels.isin(train_groups)].to_numpy()
    return SplitIndices(
        train=train,
        calib=calib,
        test=test,
        split_name=split_name,
        test_is_ood=np.ones(len(test), dtype=bool),
    )


def balanced_group_fold_split(
    df: pd.DataFrame,
    group_col: str,
    split_name: str,
    fold_index: int,
    *,
    n_splits: int = 5,
    fold_seed: int = 42,
    calib_seed: int = 42,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    """Build a row-balanced test fold while keeping every group intact."""
    if group_col not in df.columns:
        raise KeyError(f"Missing grouping column '{group_col}'.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if not 0 <= fold_index < n_splits:
        raise ValueError("fold_index must be between 0 and n_splits - 1.")

    labels = df[group_col].fillna("__missing_group__").astype(str)
    group_sizes = labels.groupby(labels, dropna=False).size()
    groups = group_sizes.index.to_list()
    if len(groups) < n_splits + 1:
        raise ValueError(
            "Balanced grouped folds require more groups than test folds."
        )

    rng = _rng(fold_seed)
    rng.shuffle(groups)
    groups.sort(key=lambda group: int(group_sizes.loc[group]), reverse=True)
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_rows = np.zeros(n_splits, dtype=int)
    for group in groups:
        target_fold = int(np.argmin(fold_rows))
        fold_groups[target_fold].append(group)
        fold_rows[target_fold] += int(group_sizes.loc[group])

    test_groups = fold_groups[fold_index]
    remaining_groups = [
        group
        for index, assigned in enumerate(fold_groups)
        if index != fold_index
        for group in assigned
    ]

    # Exact subset-sum selection keeps the calibration row count close to its
    # target without splitting a reference group.
    calib_rng = _rng(calib_seed)
    calib_rng.shuffle(remaining_groups)
    remaining_rows = int(group_sizes.loc[remaining_groups].sum())
    calib_target = max(1, int(round(remaining_rows * calib_fraction)))
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group in remaining_groups:
        size = int(group_sizes.loc[group])
        for total, selected in sorted(
            list(reachable.items()),
            reverse=True,
        ):
            candidate = total + size
            if candidate <= calib_target and candidate not in reachable:
                reachable[candidate] = (*selected, group)
    best_total = max(reachable)
    if best_total == 0:
        calib_groups = [
            min(
                remaining_groups,
                key=lambda group: int(group_sizes.loc[group]),
            )
        ]
    else:
        calib_groups = list(reachable[best_total])
    calib_group_set = set(calib_groups)
    train_groups = [
        group for group in remaining_groups if group not in calib_group_set
    ]

    test = df.index[labels.isin(test_groups)].to_numpy()
    calib = df.index[labels.isin(calib_groups)].to_numpy()
    train = df.index[labels.isin(train_groups)].to_numpy()
    return SplitIndices(
        train=train,
        calib=calib,
        test=test,
        split_name=split_name,
        test_is_ood=np.ones(len(test), dtype=bool),
    )


def group_holdout_split(
    df: pd.DataFrame,
    group_col: str,
    split_name: str,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    rng = _rng(seed)
    group_sizes = df.groupby(group_col, dropna=False).size().sort_values(ascending=False)
    groups = group_sizes.index.to_list()
    rng.shuffle(groups)
    held_out: list[str] = []
    held_out_rows = 0
    target_rows = max(1, int(round(len(df) * holdout_fraction)))
    for group in groups:
        held_out.append(group)
        held_out_rows += int(group_sizes.loc[group])
        if held_out_rows >= target_rows:
            break
    test_mask = df[group_col].isin(held_out).to_numpy()
    test = df.index.to_numpy()[test_mask]
    remaining = df.index.to_numpy()[~test_mask]
    train, calib = _partition(remaining, rng, calib_fraction)
    return SplitIndices(
        train=train,
        calib=calib,
        test=test,
        split_name=split_name,
        test_is_ood=np.ones(len(test), dtype=bool),
    )


def time_split(
    df: pd.DataFrame,
    year_col: str,
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    if year_col not in df.columns:
        raise KeyError(f"Missing temporal column '{year_col}'.")
    ordered = df[[year_col]].copy()
    ordered[year_col] = pd.to_numeric(ordered[year_col], errors="coerce")
    ordered = ordered.sort_values(year_col)
    n_test = max(1, int(round(len(df) * holdout_fraction)))
    test = ordered.tail(n_test).index.to_numpy()
    remaining = ordered.head(len(df) - n_test).index.to_numpy()
    train, calib = _partition(remaining, _rng(seed), calib_fraction)
    return SplitIndices(
        train=train,
        calib=calib,
        test=test,
        split_name="temporal",
        test_is_ood=np.ones(len(test), dtype=bool),
    )


def species_holdout_split(
    df: pd.DataFrame,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    level: str = "species",
    seed: int = 42,
    holdout_fraction: float = 0.2,
    calib_fraction: float = 0.125,
) -> SplitIndices:
    if not hasattr(schema, level):
        raise KeyError(f"Unknown taxonomy level '{level}'.")
    return group_holdout_split(
        df,
        group_col=getattr(schema, level),
        split_name=f"{level}_holdout",
        seed=seed,
        holdout_fraction=holdout_fraction,
        calib_fraction=calib_fraction,
    )


def named_chemical_class_holdout_split(
    df: pd.DataFrame,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    seed: int = 42,
    calib_fraction: float = 0.125,
    held_out_class: str | None = None,
) -> SplitIndices:
    """Leave out one fixed atomic class, including all multilabel cases."""
    labels = df[schema.chemical_class].fillna("unclassified").astype(str).str.strip()
    class_tokens = labels.map(
        lambda value: {
            token.strip()
            for token in value.split(";")
            if token.strip() and token.strip().casefold() != "unclassified"
        }
    )
    held_out_class = held_out_class or named_class_for_seed(seed)
    test_mask = class_tokens.map(lambda tokens: held_out_class in tokens)
    if not test_mask.any():
        raise ValueError(
            f"Chemical-class holdout contains no rows for {held_out_class!r}."
        )
    rng = _rng(seed)
    test = df.index.to_numpy()[test_mask.to_numpy()]
    remaining = df.index.to_numpy()[~test_mask.to_numpy()]
    train, calib = _partition(remaining, rng, calib_fraction)
    return SplitIndices(
        train=train,
        calib=calib,
        test=test,
        split_name="chemical_class",
        test_is_ood=np.ones(len(test), dtype=bool),
    )


def named_class_for_seed(seed: int) -> str:
    index = (int(seed) - NAMED_CLASS_SEED_OFFSET) % len(NAMED_CLASS_HOLDOUTS)
    return NAMED_CLASS_HOLDOUTS[index]


def build_split(
    df: pd.DataFrame,
    split: str,
    schema: EcoOODSchema = DEFAULT_SCHEMA,
    seed: int = 42,
) -> SplitIndices:
    if split == "random":
        return random_split(df, seed=seed)
    if split == "scaffold":
        return scaffold_split(df, schema=schema, seed=seed)
    if split == "scaffold_acyclic_identity":
        return scaffold_acyclic_identity_split(df, schema=schema, seed=seed)
    if split == "chemical_random":
        return chemical_random_split(df, schema=schema, seed=seed)
    if split == "chemical_class":
        return named_chemical_class_holdout_split(df, schema=schema, seed=seed)
    if split == "species":
        return species_holdout_split(df, schema=schema, level="species", seed=seed)
    if split == "genus":
        return species_holdout_split(df, schema=schema, level="genus", seed=seed)
    if split == "temporal":
        return time_split(df, year_col=schema.study_year, seed=seed)
    raise ValueError(f"Unsupported split '{split}'.")
