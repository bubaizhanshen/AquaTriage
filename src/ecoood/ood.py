from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import sparse
from sklearn.covariance import LedoitWolf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from .features import BIOACTIVITY_AUDIT_FIELDS, FeatureBundle
from .schema import DEFAULT_SCHEMA, EcoOODSchema


REVISED_AXIS_COMPONENTS = {
    "chemical": ("d_chem_knn", "d_chem_mahal"),
    "biological": ("d_species_tax",),
    "contextual": ("d_context", "context_missing_fraction"),
    "bioactivity": ("d_mech", "bioactivity_missing_fraction"),
    "uncertainty": ("u_model",),
}

LEGACY_AXIS_COMPONENTS = {
    "chemical": ("d_chem_knn", "d_chem_mahal"),
    "biological": ("d_species_knn", "d_species_tax"),
    "contextual": ("d_context",),
    "bioactivity": ("d_mech",),
    "uncertainty": ("u_model",),
}

# Backward-compatible alias for downstream imports that use the revised default.
AXIS_COMPONENTS = REVISED_AXIS_COMPONENTS


def _mean_knn_distance(
    train: np.ndarray,
    query: np.ndarray,
    metric: str = "euclidean",
    n_neighbors: int = 5,
) -> np.ndarray:
    if train.shape[1] == 0:
        return np.zeros(query.shape[0], dtype=float)
    k = max(1, min(n_neighbors, train.shape[0]))
    nn = NearestNeighbors(metric=metric, n_neighbors=k)
    nn.fit(train)
    distances, _ = nn.kneighbors(query)
    return distances.mean(axis=1)


def _first_group_indices(groups) -> np.ndarray:
    values = pd.Series(np.asarray(groups)).astype("string").fillna("__missing__")
    return np.sort(values.drop_duplicates(keep="first").index.to_numpy(dtype=int))


def _mean_distinct_group_knn_distance(
    train,
    query,
    train_groups,
    *,
    metric: str = "euclidean",
    n_neighbors: int = 5,
) -> np.ndarray:
    indices = _first_group_indices(train_groups)
    return _mean_knn_distance(
        train[indices],
        query,
        metric=metric,
        n_neighbors=n_neighbors,
    )


def _mean_tanimoto_knn_distance(
    train,
    query,
    train_groups,
    *,
    n_neighbors: int = 5,
    chunk_size: int = 256,
) -> np.ndarray:
    """Mean distance to the nearest distinct chemicals by Tanimoto similarity."""
    indices = _first_group_indices(train_groups)
    train_csr = sparse.csr_matrix(train[indices], dtype=np.float32)
    query_csr = sparse.csr_matrix(query, dtype=np.float32)
    if train_csr.shape[0] == 0:
        return np.ones(query_csr.shape[0], dtype=float)
    k = min(max(1, n_neighbors), train_csr.shape[0])
    train_counts = np.asarray(train_csr.sum(axis=1)).ravel()
    scores = np.empty(query_csr.shape[0], dtype=float)
    for start in range(0, query_csr.shape[0], chunk_size):
        stop = min(start + chunk_size, query_csr.shape[0])
        block = query_csr[start:stop]
        intersections = (block @ train_csr.T).toarray()
        query_counts = np.asarray(block.sum(axis=1)).ravel()
        unions = query_counts[:, None] + train_counts[None, :] - intersections
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        distances = 1.0 - similarities
        nearest = np.partition(distances, k - 1, axis=1)[:, :k]
        scores[start:stop] = nearest.mean(axis=1)
    return scores


def _mahalanobis_scores(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    if train.shape[1] == 0:
        return np.zeros(len(query), dtype=float)
    centered = train - train.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov += np.eye(cov.shape[0]) * 1e-6
    inv = np.linalg.pinv(cov)
    delta = query - train.mean(axis=0, keepdims=True)
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, inv, delta))


def _fit_shrinkage_mahalanobis(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] == 0:
        return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)
    estimator = LedoitWolf().fit(values)
    return np.asarray(estimator.location_, dtype=float), np.asarray(estimator.precision_, dtype=float)


def _mahalanobis_from_fit(
    query: np.ndarray,
    location: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    values = np.asarray(query, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)
    delta = values - location[None, :]
    return np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, precision, delta)))


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(df), 0), dtype=float)
    return np.column_stack(
        [pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float) for column in columns]
    )


def _context_columns(
    df: pd.DataFrame,
    schema: EcoOODSchema,
    *,
    include_study_year: bool = False,
) -> tuple[list[str], list[str]]:
    numeric = [
        column
        for column in (
            schema.duration_h,
            schema.temperature_c,
            schema.ph,
            *([schema.study_year] if include_study_year else []),
            *sorted(column for column in df.columns if column.startswith("ctx_")),
        )
        if column in df.columns
    ]
    categorical = [
        column for column in (schema.medium,) if column in df.columns
    ]
    return list(dict.fromkeys(numeric)), categorical


def _endpoint_gower_knn_distance(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    schema: EcoOODSchema,
    *,
    n_neighbors: int = 5,
    include_study_year: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Gower kNN within endpoint, plus the query's missing context fraction."""
    numeric_columns, categorical_columns = _context_columns(
        train_df,
        schema,
        include_study_year=include_study_year,
    )
    train_numeric = _numeric_frame(train_df, numeric_columns)
    query_numeric = _numeric_frame(query_df, numeric_columns)
    if train_numeric.shape[1]:
        ranges = np.nanmax(train_numeric, axis=0) - np.nanmin(train_numeric, axis=0)
        ranges[~np.isfinite(ranges) | (ranges <= 1e-12)] = 1.0
        missing_fraction = np.mean(~np.isfinite(query_numeric), axis=1)
    else:
        ranges = np.ones(0, dtype=float)
        missing_fraction = np.zeros(len(query_df), dtype=float)

    train_categorical = {
        column: train_df[column].astype("string").to_numpy()
        for column in categorical_columns
    }
    query_categorical = {
        column: query_df[column].astype("string").to_numpy()
        for column in categorical_columns
    }
    train_endpoint = train_df[schema.endpoint].astype("string").to_numpy()
    query_endpoint = query_df[schema.endpoint].astype("string").to_numpy()
    scores = np.ones(len(query_df), dtype=float)

    for row in range(len(query_df)):
        candidate = np.flatnonzero(train_endpoint == query_endpoint[row])
        if candidate.size == 0:
            continue
        numerator = np.zeros(candidate.size, dtype=float)
        denominator = np.zeros(candidate.size, dtype=float)
        for column_index in range(train_numeric.shape[1]):
            query_value = query_numeric[row, column_index]
            train_values = train_numeric[candidate, column_index]
            valid = np.isfinite(query_value) & np.isfinite(train_values)
            if not np.any(valid):
                continue
            difference = np.minimum(
                np.abs(train_values[valid] - query_value) / ranges[column_index],
                1.0,
            )
            numerator[valid] += difference
            denominator[valid] += 1.0
        for column in categorical_columns:
            query_value = query_categorical[column][row]
            train_values = train_categorical[column][candidate]
            valid = pd.notna(query_value) & pd.notna(train_values)
            if not np.any(valid):
                continue
            numerator[valid] += (train_values[valid] != query_value).astype(float)
            denominator[valid] += 1.0
        distances = np.divide(
            numerator,
            denominator,
            out=np.full(candidate.size, np.nan, dtype=float),
            where=denominator > 0,
        )
        finite = distances[np.isfinite(distances)]
        if finite.size:
            k = min(max(1, n_neighbors), finite.size)
            scores[row] = float(np.partition(finite, k - 1)[:k].mean())
    return scores, np.asarray(missing_fraction, dtype=float)


def _bioactivity_knn_distance(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    schema: EcoOODSchema,
    *,
    n_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    columns = sorted(
        column
        for column in train_df.columns
        if (
            column.startswith("mech_")
            and column not in BIOACTIVITY_AUDIT_FIELDS
            and pd.api.types.is_numeric_dtype(train_df[column])
        )
    )
    if not columns:
        return np.zeros(len(query_df), dtype=float), np.ones(len(query_df), dtype=float)
    train = _numeric_frame(train_df, columns)
    query = _numeric_frame(query_df, columns)
    missing_fraction = np.mean(~np.isfinite(query), axis=1)
    unique_indices = _first_group_indices(train_df[schema.chemical_id])
    train = train[unique_indices]

    means = np.nanmean(train, axis=0)
    means[~np.isfinite(means)] = 0.0
    scales = np.nanstd(train, axis=0, ddof=0)
    scales[~np.isfinite(scales) | (scales <= 1e-12)] = 1.0
    train_scaled = (train - means) / scales
    query_scaled = (query - means) / scales
    distances = nan_euclidean_distances(query_scaled, train_scaled)
    scores = np.zeros(len(query_df), dtype=float)
    for row, values in enumerate(distances):
        finite = values[np.isfinite(values)]
        if finite.size:
            k = min(max(1, n_neighbors), finite.size)
            scores[row] = float(np.partition(finite, k - 1)[:k].mean())
    return scores, np.asarray(missing_fraction, dtype=float)


def _taxon_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().casefold()
    return text or None


def _taxonomy_novelty(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    schema: EcoOODSchema,
) -> np.ndarray:
    """Return normalized lineage distance to the deepest training-supported rank.

    A query receives zero when its full species lineage is represented in the
    training archive. An unseen species in a seen genus receives 1/6, whereas a
    lineage with no represented phylum receives 1. This preserves information
    about higher-rank support instead of assigning every unseen species the same
    maximum novelty.
    """
    levels = [
        schema.phylum,
        schema.clazz,
        schema.order,
        schema.family,
        schema.genus,
        schema.species,
    ]
    available_levels = [level for level in levels if level in train_df and level in query_df]
    if not available_levels:
        return np.zeros(len(query_df), dtype=float)

    seen_prefixes: dict[int, set[tuple[str, ...]]] = {
        depth: set() for depth in range(1, len(available_levels) + 1)
    }
    for _, row in train_df.iterrows():
        lineage: list[str] = []
        for depth, level in enumerate(available_levels, start=1):
            value = _taxon_value(row.get(level))
            if value is None:
                break
            lineage.append(value)
            seen_prefixes[depth].add(tuple(lineage))

    scores = np.zeros(len(query_df), dtype=float)
    for i, (_, row) in enumerate(query_df.iterrows()):
        lineage: list[str] = []
        deepest_supported = 0
        for depth, level in enumerate(available_levels, start=1):
            value = _taxon_value(row.get(level))
            if value is None:
                break
            lineage.append(value)
            if tuple(lineage) in seen_prefixes[depth]:
                deepest_supported = depth
            else:
                break
        scores[i] = 1.0 - deepest_supported / len(available_levels)
    return scores


def _high_error_labels(
    residuals: np.ndarray,
    high_error_quantile: float,
    *,
    groups: pd.Series | np.ndarray | None = None,
    groupwise: bool = False,
) -> tuple[np.ndarray, float, dict[str, float]]:
    residuals = np.asarray(residuals, dtype=float)
    if not 0 < high_error_quantile < 1:
        raise ValueError("high_error_quantile must be between 0 and 1.")
    pooled_threshold = float(np.quantile(residuals, high_error_quantile))
    if groups is None or not groupwise:
        return residuals >= pooled_threshold, pooled_threshold, {}

    group_values = pd.Series(groups, dtype="object").fillna("missing").astype(str).to_numpy()
    labels = np.zeros(len(residuals), dtype=bool)
    thresholds: dict[str, float] = {}
    for group in sorted(set(group_values)):
        mask = group_values == group
        threshold = float(np.quantile(residuals[mask], high_error_quantile))
        thresholds[group] = threshold
        labels[mask] = residuals[mask] >= threshold
    return labels, pooled_threshold, thresholds


def _group_balance_weights(groups: pd.Series | np.ndarray | None) -> np.ndarray | None:
    if groups is None:
        return None
    values = pd.Series(groups, dtype="object").fillna("missing").astype(str)
    counts = values.value_counts()
    n_groups = len(counts)
    if n_groups <= 1:
        return None
    return values.map(lambda value: len(values) / (n_groups * counts[value])).to_numpy(dtype=float)


def _make_logistic_model() -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        class_weight=None,
        fit_intercept=True,
        max_iter=1000,
        random_state=0,
    )


def calibration_meta_bootstrap(
    components: pd.DataFrame,
    residuals: np.ndarray,
    chemical_ids,
    *,
    n_replicates: int = 200,
    high_error_quantile: float = 0.9,
    seed: int = 42,
) -> pd.DataFrame:
    """Assess residual-risk score stability under chemical-cluster resampling."""
    residuals = np.asarray(residuals, dtype=float)
    groups = pd.Series(np.asarray(chemical_ids)).astype("string").fillna("__missing__")
    if len(components) != len(residuals) or len(components) != len(groups):
        raise ValueError("Components, residuals, and chemical_ids must align.")
    unique_groups = groups.unique()
    if len(unique_groups) < 2:
        return pd.DataFrame()

    reference = CalibrationRiskScorer().fit(
        components,
        residuals,
        high_error_quantile=high_error_quantile,
    )
    reference_scores = reference.predict(components)
    group_rows = {
        group: np.flatnonzero(groups.to_numpy() == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | bool]] = []
    for replicate in range(n_replicates):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        sampled_rows = np.concatenate([group_rows[group] for group in sampled_groups])
        scorer = CalibrationRiskScorer().fit(
            components.iloc[sampled_rows],
            residuals[sampled_rows],
            high_error_quantile=high_error_quantile,
        )
        scores = scorer.predict(components)
        rank_correlation = spearmanr(reference_scores, scores).statistic
        row: dict[str, float | int | bool] = {
            "replicate": replicate,
            "sampled_case_n": int(len(sampled_rows)),
            "sampled_positive_n": int(scorer.positive_count_),
            "score_spearman": float(rank_correlation),
            "converged": bool(scorer.converged_),
        }
        if scorer.model is not None:
            for column, coefficient in zip(
                components.columns,
                scorer.model.coef_[0],
                strict=True,
            ):
                row[f"coef_{column}"] = float(coefficient)
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class OODComponents:
    chemical: np.ndarray
    species: np.ndarray
    context: np.ndarray
    mechanism: np.ndarray
    model_uncertainty: np.ndarray
    ecoood_score: np.ndarray


class CalibrationRiskScorer:
    """Calibration-trained high-error risk scorer for matched supervision checks.

    The scorer uses only calibration-fold features and residual labels. It is
    intentionally generic so EcoOOD can be compared with simpler risk models
    trained with the same residual-supervision budget.
    """

    def __init__(self) -> None:
        self.scaler = MinMaxScaler(clip=False)
        self.model: LogisticRegression | None = None
        self.high_error_quantile_: float | None = None
        self.high_error_threshold_: float | None = None
        self.positive_count_: int = 0
        self.positive_rate_: float = 0.0
        self.converged_: bool = True

    def fit(
        self,
        features: pd.DataFrame,
        residuals: np.ndarray,
        high_error_quantile: float = 0.9,
    ) -> "CalibrationRiskScorer":
        if features.empty:
            raise ValueError("Calibration risk features must contain at least one column.")
        residuals = np.asarray(residuals, dtype=float)
        labels, threshold, _ = _high_error_labels(residuals, high_error_quantile)
        self.high_error_quantile_ = high_error_quantile
        self.high_error_threshold_ = threshold
        return self.fit_labels(features, labels)

    def fit_labels(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
    ) -> "CalibrationRiskScorer":
        """Fit a calibration risk model to a prespecified binary outcome."""
        if features.empty:
            raise ValueError("Calibration risk features must contain at least one column.")
        labels = np.asarray(labels, dtype=bool)
        if len(labels) != len(features):
            raise ValueError("Calibration labels must align with the feature rows.")
        self.positive_count_ = int(labels.sum())
        self.positive_rate_ = float(labels.mean())
        self.scaler.fit(features)
        scaled = self.scaler.transform(features)
        if len(np.unique(labels)) > 1:
            self.model = _make_logistic_model()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                self.model.fit(scaled, labels.astype(int))
            self.converged_ = not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        scaled = self.scaler.transform(features)
        if self.model is not None:
            return self.model.predict_proba(scaled)[:, 1]
        return np.full(len(features), self.positive_rate_, dtype=float)


class EcoOODScorer:
    def __init__(
        self,
        schema: EcoOODSchema = DEFAULT_SCHEMA,
        *,
        component_mode: str = "revised",
        n_neighbors: int = 5,
        fingerprint_metric: str = "tanimoto",
        include_study_year: bool = False,
    ) -> None:
        if component_mode not in {"revised", "legacy"}:
            raise ValueError("component_mode must be 'revised' or 'legacy'.")
        if fingerprint_metric not in {"tanimoto", "cosine"}:
            raise ValueError("fingerprint_metric must be 'tanimoto' or 'cosine'.")
        self.schema = schema
        self.component_mode = component_mode
        self.n_neighbors = max(1, int(n_neighbors))
        self.fingerprint_metric = fingerprint_metric
        self.include_study_year = include_study_year
        self.axis_components = (
            REVISED_AXIS_COMPONENTS if component_mode == "revised" else LEGACY_AXIS_COMPONENTS
        )
        self.train_df: pd.DataFrame | None = None
        self.train_bundle: FeatureBundle | None = None
        self.descriptor_location = np.zeros(0, dtype=float)
        self.descriptor_precision = np.zeros((0, 0), dtype=float)
        self.component_scaler = MinMaxScaler(clip=False)
        self.meta_model: LogisticRegression | None = None
        self.high_error_quantile_: float | None = None
        self.high_error_threshold_: float | None = None
        self.group_thresholds_: dict[str, float] = {}
        self.positive_count_: int = 0
        self.calibration_count_: int = 0
        self.converged_: bool = True

    def fit(self, train_df: pd.DataFrame, train_bundle: FeatureBundle) -> "EcoOODScorer":
        self.train_df = train_df.copy()
        self.train_bundle = train_bundle
        if self.component_mode == "revised":
            unique_indices = _first_group_indices(train_df[self.schema.chemical_id])
            self.descriptor_location, self.descriptor_precision = _fit_shrinkage_mahalanobis(
                np.asarray(train_bundle.descriptor, dtype=float)[unique_indices]
            )
        return self

    def component_frame(
        self,
        df: pd.DataFrame,
        bundle: FeatureBundle,
        model_std: np.ndarray,
        interval_width: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if self.train_df is None or self.train_bundle is None:
            raise RuntimeError("EcoOODScorer must be fit before use.")
        train_bundle = self.train_bundle
        if self.component_mode == "legacy":
            chem_knn = _mean_knn_distance(
                train_bundle.fingerprint,
                bundle.fingerprint,
                metric="cosine",
                n_neighbors=self.n_neighbors,
            )
            chem_mahal = _mahalanobis_scores(train_bundle.descriptor, bundle.descriptor)
            species_knn = _mean_knn_distance(
                train_bundle.species,
                bundle.species,
                n_neighbors=self.n_neighbors,
            )
            species_tax = _taxonomy_novelty(self.train_df, df, self.schema)
            context = _mean_knn_distance(
                train_bundle.context,
                bundle.context,
                n_neighbors=self.n_neighbors,
            )
            mechanism = _mean_knn_distance(
                train_bundle.mechanism,
                bundle.mechanism,
                n_neighbors=self.n_neighbors,
            )
            component_data = {
                "d_chem_knn": chem_knn,
                "d_chem_mahal": chem_mahal,
                "d_species_knn": species_knn,
                "d_species_tax": species_tax,
                "d_context": context,
                "d_mech": mechanism,
            }
        else:
            if self.fingerprint_metric == "tanimoto":
                chem_knn = _mean_tanimoto_knn_distance(
                    train_bundle.fingerprint,
                    bundle.fingerprint,
                    self.train_df[self.schema.chemical_id],
                    n_neighbors=self.n_neighbors,
                )
            else:
                chem_knn = _mean_distinct_group_knn_distance(
                    train_bundle.fingerprint,
                    bundle.fingerprint,
                    self.train_df[self.schema.chemical_id],
                    metric="cosine",
                    n_neighbors=self.n_neighbors,
                )
            chem_mahal = _mahalanobis_from_fit(
                bundle.descriptor,
                self.descriptor_location,
                self.descriptor_precision,
            )
            species_tax = _taxonomy_novelty(self.train_df, df, self.schema)
            context, context_missing = _endpoint_gower_knn_distance(
                self.train_df,
                df,
                self.schema,
                n_neighbors=self.n_neighbors,
                include_study_year=self.include_study_year,
            )
            mechanism, mechanism_missing = _bioactivity_knn_distance(
                self.train_df,
                df,
                self.schema,
                n_neighbors=self.n_neighbors,
            )
            component_data = {
                "d_chem_knn": chem_knn,
                "d_chem_mahal": chem_mahal,
                "d_species_tax": species_tax,
                "d_context": context,
                "context_missing_fraction": context_missing,
                "d_mech": mechanism,
                "bioactivity_missing_fraction": mechanism_missing,
            }
        # With scaled conformal prediction, interval width is a fold-specific
        # constant multiple of the ensemble standard deviation. Retaining both
        # would duplicate the same uncertainty signal in the meta-model.
        model_uncertainty = np.asarray(model_std, dtype=float)
        component_data["u_model"] = model_uncertainty
        return pd.DataFrame(component_data, index=df.index)

    def fit_meta(
        self,
        components: pd.DataFrame,
        residuals: np.ndarray,
        high_error_quantile: float = 0.9,
        *,
        groups: pd.Series | np.ndarray | None = None,
        groupwise_labels: bool = False,
        balance_groups: bool = False,
    ) -> "EcoOODScorer":
        labels, threshold, group_thresholds = _high_error_labels(
            residuals,
            high_error_quantile,
            groups=groups,
            groupwise=groupwise_labels,
        )
        self.high_error_quantile_ = high_error_quantile
        self.high_error_threshold_ = threshold
        self.group_thresholds_ = group_thresholds
        self.positive_count_ = int(labels.sum())
        self.calibration_count_ = int(len(labels))
        self.component_scaler.fit(components)
        scaled = self.component_scaler.transform(components)
        if len(np.unique(labels)) > 1:
            sample_weight = _group_balance_weights(groups) if balance_groups else None
            self.meta_model = _make_logistic_model()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                self.meta_model.fit(
                    scaled,
                    labels.astype(int),
                    sample_weight=sample_weight,
                )
            self.converged_ = not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
        return self

    def scaled_axis_frame(self, components: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.component_scaler, "n_features_in_"):
            raise RuntimeError("EcoOODScorer meta-model must be fit before axis scaling.")
        scaled = pd.DataFrame(
            self.component_scaler.transform(components),
            columns=components.columns,
            index=components.index,
        )
        return pd.DataFrame(
            {
                axis: scaled.loc[:, list(columns)].mean(axis=1)
                for axis, columns in self.axis_components.items()
            },
            index=components.index,
        )

    def diagnostics(self) -> dict[str, float | int | bool]:
        result: dict[str, float | int | bool] = {
            "calibration_n": self.calibration_count_,
            "high_error_n": self.positive_count_,
            "high_error_quantile": (
                float(self.high_error_quantile_)
                if self.high_error_quantile_ is not None
                else float("nan")
            ),
            "high_error_threshold": (
                float(self.high_error_threshold_)
                if self.high_error_threshold_ is not None
                else float("nan")
            ),
            "meta_converged": self.converged_,
            "meta_minmax_clip": False,
        }
        if self.meta_model is not None:
            for column, coefficient in zip(
                self.component_scaler.feature_names_in_,
                self.meta_model.coef_[0],
                strict=True,
            ):
                result[f"coef_{column}"] = float(coefficient)
            result["intercept"] = float(self.meta_model.intercept_[0])
        return result

    def score_components(self, components: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.component_scaler, "n_features_in_"):
            raise RuntimeError("EcoOODScorer meta-model must be fit before scoring.")
        scaled = self.component_scaler.transform(components)
        if self.meta_model is not None:
            return self.meta_model.predict_proba(scaled)[:, 1]
        return np.asarray(scaled.mean(axis=1), dtype=float)

    def predict(
        self,
        df: pd.DataFrame,
        bundle: FeatureBundle,
        model_std: np.ndarray,
        interval_width: np.ndarray | None = None,
    ) -> OODComponents:
        components = self.component_frame(df, bundle, model_std=model_std, interval_width=interval_width)
        ecoood_score = self.score_components(components)
        axes = self.scaled_axis_frame(components)
        return OODComponents(
            chemical=axes["chemical"].to_numpy(),
            species=axes["biological"].to_numpy(),
            context=axes["contextual"].to_numpy(),
            mechanism=axes["bioactivity"].to_numpy(),
            model_uncertainty=axes["uncertainty"].to_numpy(),
            ecoood_score=np.asarray(ecoood_score, dtype=float),
        )
