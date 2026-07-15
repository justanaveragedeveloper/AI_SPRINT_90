import asyncio
import pytest
from pydantic import ValidationError
from day11_drift_monitor import StatisticalHeartMonitor, FeatureVectorBatch

@pytest.mark.anyio
async def test_normal_flow():
    m = StatisticalHeartMonitor([10,5],[1,1])
    r = await m.monitor_stream_batch(
        FeatureVectorBatch(tenant_id="abc", vectors=[[10,5],[11,5]])
    )
    assert r.processed_samples == 2

def test_nan_blocked():
    with pytest.raises(ValidationError):
        FeatureVectorBatch(tenant_id="abc", vectors=[[1,float("nan")]])

def test_baseline_validation():
    with pytest.raises(ValueError):
        StatisticalHeartMonitor([1,2],[1])

@pytest.mark.anyio
async def test_tenant_isolation():
    m = StatisticalHeartMonitor([0],[1])
    await m.monitor_stream_batch(FeatureVectorBatch(tenant_id="A12", vectors=[[1]]))
    await m.monitor_stream_batch(FeatureVectorBatch(tenant_id="B12", vectors=[[2]]))
    assert len(m._tenant_states) == 2

@pytest.mark.anyio
async def test_concurrent_updates():
    m = StatisticalHeartMonitor([0],[1])
    payload = FeatureVectorBatch(tenant_id="abc", vectors=[[1],[2],[3]])
    await asyncio.gather(*[m.monitor_stream_batch(payload) for _ in range(20)])
    assert m._tenant_states["abc"].count == 60
