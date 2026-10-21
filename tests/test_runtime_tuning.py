"""CPU-only default/guard/affinity intersection checks; never bind this process."""
from pathlib import Path

import pytest

from humanoidverse import distributed_precision as bucket
from humanoidverse import runtime_cpu as numa


def test_binding_default_off_has_no_os_side_effect(monkeypatch):
    monkeypatch.delenv("UFO_CPU_NUMA_BIND", raising=False)
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda *_: pytest.fail("default-off read affinity"))
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda *_: pytest.fail("default-off set affinity"))
    assert numa.bind_worker_cpu_numa(0, 8) is None


def test_binding_uses_correct_node_and_only_allowed_cpus(monkeypatch):
    monkeypatch.setenv("UFO_CPU_NUMA_BIND", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    seen, changed = [], []

    def read(path):
        seen.append(str(path))
        return "0-3,8-11" if "node0/" in str(path) else "4-7,12-15"

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda _: {1, 3, 4, 7, 9, 12})
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda pid, cpus: changed.append((pid, cpus)))
    assert numa.bind_worker_cpu_numa(2, 8) == {"local_rank": 2, "numa_node": 0, "cpu_affinity": [1, 3, 9]}
    assert numa.bind_worker_cpu_numa(6, 8) == {"local_rank": 6, "numa_node": 1, "cpu_affinity": [4, 7, 12]}
    assert seen == ["/sys/devices/system/node/node0/cpulist", "/sys/devices/system/node/node1/cpulist"]
    assert changed == [(0, {1, 3, 9}), (0, {4, 7, 12})]


def test_binding_rejects_wrong_profile_and_empty_intersection(monkeypatch):
    monkeypatch.setenv("UFO_CPU_NUMA_BIND", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7,0,1,2,3")
    monkeypatch.setattr(numa.os, "sched_setaffinity", lambda *_: pytest.fail("invalid config set affinity"))
    with pytest.raises(ValueError, match="verified"):
        numa.bind_worker_cpu_numa(0, 8)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    with pytest.raises(ValueError, match="verified"):
        numa.bind_worker_cpu_numa(0, 4)
    monkeypatch.setattr(Path, "read_text", lambda _: "0-3")
    monkeypatch.setattr(numa.os, "sched_getaffinity", lambda _: {8, 9})
    with pytest.raises(ValueError, match="no CPUs allowed"):
        numa.bind_worker_cpu_numa(0, 8)
    monkeypatch.setenv("UFO_CPU_NUMA_BIND", "true")
    with pytest.raises(ValueError, match="0 or 1"):
        numa.bind_worker_cpu_numa(0, 8)


def test_bucket_default_preserves_existing_override_and_reads_each_call(monkeypatch):
    monkeypatch.delenv("UFO_FP16_BUCKET_MB", raising=False)
    monkeypatch.setattr(bucket, "_MAX_BUCKET_BYTES", 128)
    assert bucket._configured_bucket_bytes() == 128
    monkeypatch.setenv("UFO_FP16_BUCKET_MB", "16")
    assert bucket._configured_bucket_bytes() == 16 * 1024 * 1024
    monkeypatch.setenv("UFO_FP16_BUCKET_MB", "64")
    assert bucket._configured_bucket_bytes() == 64 * 1024 * 1024


@pytest.mark.parametrize("value", ["0", "-1", "65", "nan", "16.5", ""])
def test_invalid_bucket_bounds(monkeypatch, value):
    monkeypatch.setenv("UFO_FP16_BUCKET_MB", value)
    with pytest.raises(ValueError, match="integer from 1 to 64"):
        bucket._configured_bucket_bytes()
