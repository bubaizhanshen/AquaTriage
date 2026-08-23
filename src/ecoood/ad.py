from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .features import FeatureBundle
from .ood import _mahalanobis_scores, _mean_knn_distance


def _mean_distinct_group_knn_distance(
    train,
    query,
    train_groups,
    *,
    n_neighbors: int = 5,
    metric: str = "euclidean",
) -> np.ndarray:
    """Average group-minimum distances so each chemical contributes once."""
    group_values = pd.Series(np.asarray(train_groups)).astype("string").fillna("__missing__")
    group_codes, uniques = pd.factorize(group_values, sort=False)
    if len(uniques) == 0:
        return np.full(query.shape[0], np.nan, dtype=float)
    k = min(max(1, n_neighbors), len(uniques))
    scores: list[float] = []
    for distance_chunk in pairwise_distances_chunked(query, train, metric=metric):
        for distances in distance_chunk:
            group_minimum = np.full(len(uniques), np.inf, dtype=float)
            np.minimum.at(group_minimum, group_codes, distances)
            nearest = np.partition(group_minimum, k - 1)[:k]
            scores.append(float(np.mean(nearest)))
    return np.asarray(scores, dtype=float)


def _tanimoto_novelty(train, query, chunk_size: int = 256) -> np.ndarray:
    """Return 1 - maximum training-set Tanimoto similarity for binary fingerprints."""
    train_csr = sparse.csr_matrix(train, dtype=np.float32)
    query_csr = sparse.csr_matrix(query, dtype=np.float32)
    if train_csr.shape[0] == 0:
        return np.ones(query_csr.shape[0], dtype=float)

    train_counts = np.asarray(train_csr.sum(axis=1)).ravel()
    novelty = np.empty(query_csr.shape[0], dtype=float)
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
        novelty[start:stop] = 1.0 - similarities.max(axis=1)
    return novelty


@dataclass
class ADBaselineScores:
    similarity: np.ndarray
    leverage: np.ndarray
    descriptor_range: np.ndarray
    distance_to_model: np.ndarray
    equal_block_distance: np.ndarray
    equal_block_distance_distinct_chemical: np.ndarray
    interval_width: np.ndarray
    mahalanobis: np.ndarray
    isolation_forest: np.ndarray
    lof: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "ad_similarity": self.similarity,
            "ad_leverage": self.leverage,
            "ad_range": self.descriptor_range,
            "ad_distance_to_model": self.distance_to_model,
            "ad_equal_block_distance": self.equal_block_distance,
            "ad_equal_block_distance_distinct_chemical": (
                self.equal_block_distance_distinct_chemical
            ),
            "uncertainty_interval_width": self.interval_width,
            "ood_mahalanobis": self.mahalanobis,
            "ood_isolation_forest": self.isolation_forest,
            "ood_lof": self.lof,
        }


class ApplicabilityDomainScorer:
    def __init__(self) -> None:
        self.train_bundle: FeatureBundle | None = None
        self.descriptor_min: np.ndarray | None = None
        self.descriptor_max: np.ndarray | None = None
        self.xtx_inv: np.ndarray | None = None
        self.dense_scaler: StandardScaler | None = None
        self.dense_train: np.ndarray | None = None
        self.isolation_forest: IsolationForest | None = None
        self.lof: LocalOutlierFactor | None = None
        self.equal_block_scales: list[float] = []
        self.equal_block_scale_by_name: dict[str, float] = {}
        self.equal_block_train: sparse.csr_matrix | None = None
        self.train_chemical_ids: np.ndarray | None = None

    @staticmethod
    def _as_csr(block) -> sparse.csr_matrix:
        if sparse.issparse(block):
            return sparse.csr_matrix(block, dtype=np.float32)
        values = np.asarray(block, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return sparse.csr_matrix(values)

    @classmethod
    def _support_blocks(
        cls,
        bundle: FeatureBundle,
    ) -> list[tuple[str, sparse.csr_matrix]]:
        return [
            ("fingerprint", cls._as_csr(bundle.fingerprint)),
            ("descriptor", cls._as_csr(bundle.descriptor)),
            ("species", cls._as_csr(bundle.species)),
            ("context", cls._as_csr(bundle.context)),
            ("bioactivity_proxy", cls._as_csr(bundle.mechanism)),
        ]

    @staticmethod
    def _block_rms_norm(block: sparse.csr_matrix) -> float:
        if block.shape[1] == 0:
            return 1.0
        squared_norm = np.asarray(block.multiply(block).sum(axis=1)).ravel()
        value = float(np.sqrt(np.mean(squared_norm)))
        return value if np.isfinite(value) and value > 1e-12 else 1.0

    def _fit_equal_block_representation(
        self,
        train_bundle: FeatureBundle,
    ) -> sparse.csr_matrix:
        named_blocks = [
            (name, block)
            for name, block in self._support_blocks(train_bundle)
            if block.shape[1] > 0
        ]
        blocks = [block for _, block in named_blocks]
        self.equal_block_scales = [self._block_rms_norm(block) for block in blocks]
        self.equal_block_scale_by_name = {
            name: scale
            for (name, _), scale in zip(named_blocks, self.equal_block_scales, strict=True)
        }
        if not blocks:
            return sparse.csr_matrix((train_bundle.full.shape[0], 0), dtype=np.float32)
        normalizer = np.sqrt(len(blocks))
        scaled = [
            block.multiply(1.0 / (scale * normalizer))
            for block, scale in zip(blocks, self.equal_block_scales, strict=True)
        ]
        return sparse.hstack(scaled, format="csr")

    def _transform_equal_block_representation(
        self,
        bundle: FeatureBundle,
    ) -> sparse.csr_matrix:
        blocks = [
            block
            for _, block in self._support_blocks(bundle)
            if block.shape[1] > 0
        ]
        if len(blocks) != len(self.equal_block_scales):
            raise ValueError("Training and query feature blocks do not match.")
        if not blocks:
            return sparse.csr_matrix((bundle.full.shape[0], 0), dtype=np.float32)
        normalizer = np.sqrt(len(blocks))
        scaled = [
            block.multiply(1.0 / (scale * normalizer))
            for block, scale in zip(blocks, self.equal_block_scales, strict=True)
        ]
        return sparse.hstack(scaled, format="csr")

    @staticmethod
    def _stack_dense(bundle: FeatureBundle) -> np.ndarray:
        def _as_dense(block) -> np.ndarray:
            if sparse is not None and sparse.issparse(block):
                return block.toarray()
            return np.asarray(block, dtype=float)

        # Taxonomic one-hot support is scored separately by EcoOOD; generic dense
        # baselines use the continuous descriptor, context, and bioactivity blocks.
        blocks = [
            _as_dense(bundle.descriptor),
            _as_dense(bundle.context),
            _as_dense(bundle.mechanism),
        ]
        dense = np.hstack(blocks)
        if dense.ndim == 1:
            dense = dense.reshape(-1, 1)
        return np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(
        self,
        train_bundle: FeatureBundle,
        *,
        chemical_ids=None,
    ) -> "ApplicabilityDomainScorer":
        self.train_bundle = train_bundle
        if chemical_ids is not None:
            values = np.asarray(chemical_ids)
            if len(values) != train_bundle.full.shape[0]:
                raise ValueError("chemical_ids must align with the training feature rows.")
            self.train_chemical_ids = values
        self.equal_block_train = self._fit_equal_block_representation(train_bundle)
        descriptor = np.asarray(train_bundle.descriptor, dtype=float)
        if descriptor.ndim == 1:
            descriptor = descriptor.reshape(-1, 1)
        if descriptor.shape[1] == 0:
            self.descriptor_min = np.zeros(1, dtype=float)
            self.descriptor_max = np.zeros(1, dtype=float)
            self.xtx_inv = np.zeros((1, 1), dtype=float)
            return self

        self.descriptor_min = descriptor.min(axis=0)
        self.descriptor_max = descriptor.max(axis=0)
        design = np.hstack([np.ones((descriptor.shape[0], 1), dtype=float), descriptor])
        xtx = design.T @ design
        xtx += np.eye(xtx.shape[0]) * 1e-8
        self.xtx_inv = np.linalg.pinv(xtx)

        dense = self._stack_dense(train_bundle)
        self.dense_train = dense
        self.dense_scaler = StandardScaler()
        dense_scaled = self.dense_scaler.fit_transform(dense)
        self.isolation_forest = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42,
        ).fit(dense_scaled)
        n_neighbors = max(2, min(35, len(dense_scaled) - 1))
        if len(dense_scaled) > 2:
            self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
            self.lof.fit(dense_scaled)
        return self

    def _descriptor_range_score(self, descriptor: np.ndarray) -> np.ndarray:
        if self.descriptor_min is None or self.descriptor_max is None:
            return np.zeros(len(descriptor), dtype=float)
        if descriptor.shape[1] == 0:
            return np.zeros(len(descriptor), dtype=float)
        out_of_range = (descriptor < self.descriptor_min) | (descriptor > self.descriptor_max)
        return out_of_range.mean(axis=1).astype(float)

    def _leverage_score(self, descriptor: np.ndarray) -> np.ndarray:
        if self.xtx_inv is None or descriptor.shape[1] == 0:
            return np.zeros(len(descriptor), dtype=float)
        design = np.hstack([np.ones((descriptor.shape[0], 1), dtype=float), descriptor])
        return np.einsum("ij,jk,ik->i", design, self.xtx_inv, design)

    def predict(
        self,
        bundle: FeatureBundle,
        model_std: np.ndarray,
        interval_width: np.ndarray | None = None,
    ) -> ADBaselineScores:
        if self.train_bundle is None:
            raise RuntimeError("ApplicabilityDomainScorer must be fit before use.")
        descriptor = np.asarray(bundle.descriptor, dtype=float)
        if descriptor.ndim == 1:
            descriptor = descriptor.reshape(-1, 1)
        dense = self._stack_dense(bundle)
        similarity = _tanimoto_novelty(self.train_bundle.fingerprint, bundle.fingerprint)
        leverage = self._leverage_score(descriptor)
        descriptor_range = self._descriptor_range_score(descriptor)
        distance_to_model = _mean_knn_distance(
            self.train_bundle.full,
            bundle.full,
            metric="euclidean",
        )
        equal_block_query = self._transform_equal_block_representation(bundle)
        equal_block_distance = (
            _mean_knn_distance(
                self.equal_block_train,
                equal_block_query,
                metric="euclidean",
            )
            if self.equal_block_train is not None and self.equal_block_train.shape[1] > 0
            else np.zeros(len(distance_to_model), dtype=float)
        )
        equal_block_distance_distinct_chemical = (
            _mean_distinct_group_knn_distance(
                self.equal_block_train,
                equal_block_query,
                self.train_chemical_ids,
                n_neighbors=5,
                metric="euclidean",
            )
            if self.equal_block_train is not None
            and self.equal_block_train.shape[1] > 0
            and self.train_chemical_ids is not None
            else np.asarray(equal_block_distance, dtype=float).copy()
        )
        interval_width_arr = (
            np.asarray(interval_width, dtype=float)
            if interval_width is not None
            else np.zeros(len(distance_to_model), dtype=float)
        )
        mahalanobis = (
            _mahalanobis_scores(self.dense_train, dense)
            if self.dense_train is not None and dense.shape[1] > 0
            else np.zeros(len(distance_to_model), dtype=float)
        )
        if self.dense_scaler is not None:
            dense_scaled = self.dense_scaler.transform(dense)
        else:
            dense_scaled = dense
        isolation_forest = (
            -self.isolation_forest.score_samples(dense_scaled)
            if self.isolation_forest is not None
            else np.zeros(len(distance_to_model), dtype=float)
        )
        lof = (
            -self.lof.score_samples(dense_scaled)
            if self.lof is not None
            else np.zeros(len(distance_to_model), dtype=float)
        )
        return ADBaselineScores(
            similarity=np.asarray(similarity, dtype=float),
            leverage=np.asarray(leverage, dtype=float),
            descriptor_range=np.asarray(descriptor_range, dtype=float),
            distance_to_model=distance_to_model,
            equal_block_distance=np.asarray(equal_block_distance, dtype=float),
            equal_block_distance_distinct_chemical=np.asarray(
                equal_block_distance_distinct_chemical,
                dtype=float,
            ),
            interval_width=interval_width_arr,
            mahalanobis=np.asarray(mahalanobis, dtype=float),
            isolation_forest=np.asarray(isolation_forest, dtype=float),
            lof=np.asarray(lof, dtype=float),
        )
