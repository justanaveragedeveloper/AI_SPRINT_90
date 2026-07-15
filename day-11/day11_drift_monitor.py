"""
day11_drift_monitor_v2.py
Production-oriented streaming drift monitor with thread safety and bounded memory.
"""

import hashlib
import logging
import threading
import time
from functools import partial
from typing import Dict, List

import anyio
import numpy as np
from cachetools import TTLCache
from pydantic import BaseModel, Field, field_validator

MAX_DIMENSIONALITY = 2048
MIN_DIMENSIONALITY = 1
MAX_BATCH_SIZE = 5000
EPSILON_SAFETY = 1e-12

logger = logging.getLogger("drift_monitor")


class FeatureVectorBatch(BaseModel):
    tenant_id: str = Field(..., min_length=3)
    vectors: List[List[float]]

    @field_validator("vectors")
    @classmethod
    def validate_vectors(cls, v):
        if not v:
            raise ValueError("Empty batch")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError("Batch too large")
        dim = len(v[0])
        if not MIN_DIMENSIONALITY <= dim <= MAX_DIMENSIONALITY:
            raise ValueError("Invalid dimensionality")
        for row in v:
            if len(row) != dim:
                raise ValueError("Inconsistent dimensions")
            if not all(np.isfinite(x) for x in row):
                raise ValueError("NaN/Inf detected")
        return v


class DriftMetricsSummary(BaseModel):
    tenant_id: str
    processed_samples: int
    mean_drift_detected: bool
    max_z_score: float
    current_stream_means: List[float]
    current_stream_variances: List[float]


class OnlineWelfordMatrixState:
    def __init__(self, dimension: int):
        self.lock = threading.Lock()
        self.dim = dimension
        self.count = 0
        self.mean = np.zeros(dimension)
        self.M2 = np.zeros(dimension)

    def update_batch(self, matrix):
        with self.lock:
            for row in matrix:
                self.count += 1
                delta = row - self.mean
                self.mean += delta / self.count
                delta2 = row - self.mean
                self.M2 += delta * delta2

    @property
    def variance(self):
        if self.count < 2:
            return np.full(self.dim, EPSILON_SAFETY)
        return np.maximum(self.M2 / (self.count - 1), EPSILON_SAFETY)


class PipelineDisruptionException(Exception):
    pass


class StatisticalHeartMonitor:
    def __init__(self, baseline_mean, baseline_variance, drift_threshold_z=3.0):
        if len(baseline_mean) != len(baseline_variance):
            raise ValueError("Baseline mismatch")
        if len(baseline_mean) == 0:
            raise ValueError("Empty baseline")
        if np.any(np.array(baseline_variance) < 0):
            raise ValueError("Negative variance")

        self.baseline_mean = np.array(baseline_mean, dtype=np.float64)
        self.baseline_var = np.maximum(
            np.array(baseline_variance, dtype=np.float64), EPSILON_SAFETY
        )
        self.dim = len(baseline_mean)
        self.threshold = drift_threshold_z

        self._tenant_states = TTLCache(maxsize=10000, ttl=3600)
        self._tenant_lock = threading.Lock()
        self._semaphore = anyio.Semaphore(100)

    def _get_state(self, tenant_id):
        with self._tenant_lock:
            if tenant_id not in self._tenant_states:
                self._tenant_states[tenant_id] = OnlineWelfordMatrixState(self.dim)
            return self._tenant_states[tenant_id]

    def _compute(self, payload):
        matrix = np.array(payload.vectors, dtype=np.float64)
        if matrix.shape[1] != self.dim:
            raise ValueError("Dimension mismatch")

        state = self._get_state(payload.tenant_id)
        state.update_batch(matrix)

        z_scores = np.abs(state.mean - self.baseline_mean) / np.sqrt(self.baseline_var)
        max_z = float(np.max(z_scores))
        drift = max_z > self.threshold

        tenant_hash = hashlib.sha256(payload.tenant_id.encode()).hexdigest()[:8]
        logger.info("tenant=%s drift=%s z=%.3f", tenant_hash, drift, max_z)

        return DriftMetricsSummary(
            tenant_id=tenant_hash,
            processed_samples=state.count,
            mean_drift_detected=drift,
            max_z_score=max_z,
            current_stream_means=state.mean.tolist(),
            current_stream_variances=state.variance.tolist(),
        )

    async def monitor_stream_batch(self, payload):
        try:
            async with self._semaphore:
                return await anyio.to_thread.run_sync(
                    partial(
                    self._compute, payload
                    )
                )
        except Exception as e:
            raise PipelineDisruptionException(str(e)) from e
