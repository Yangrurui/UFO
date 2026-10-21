"""Optional CPU binding for this verified 8-GPU, two-NUMA-node profile."""
from __future__ import annotations

import os
from pathlib import Path


def bind_worker_cpu_numa(local_rank: int, world_size: int):
    enabled = os.environ.get("UFO_CPU_NUMA_BIND", "0")
    if enabled not in ("0", "1"):
        raise ValueError("UFO_CPU_NUMA_BIND must be 0 or 1")
    if enabled == "0":
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    if world_size != 8 or not 0 <= local_rank < 8 or visible != "0,1,2,3,4,5,6,7":
        raise ValueError("UFO_CPU_NUMA_BIND=1 requires the verified eight-worker GPU 0,1,2,3,4,5,6,7 profile")
    node = local_rank // 4
    cpus = set()
    for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
        bounds = [int(value) for value in part.split("-")]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    cpus &= os.sched_getaffinity(0)
    if not cpus:
        raise ValueError(f"NUMA node {node} has no CPUs allowed to worker {local_rank}")
    os.sched_setaffinity(0, cpus)
    return {"local_rank": local_rank, "numa_node": node, "cpu_affinity": sorted(cpus)}
