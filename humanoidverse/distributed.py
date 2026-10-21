from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.distributed as dist

# Scratch storage is outside model/optimizer state. Calls finish their stream
# dependency and copy results back before the next optimization phase uses it.
_BUCKET_BUFFERS: dict[tuple[str, torch.device, torch.dtype], torch.Tensor] = {}
_SYNC_BUCKET_BYTES = 64 * 1024 * 1024


def _gradient_bucket_bytes() -> int:
    megabytes = float(os.environ.get("UFO_GRAD_BUCKET_MB", "64"))
    if not math.isfinite(megabytes) or megabytes < 0:
        raise ValueError("UFO_GRAD_BUCKET_MB must be finite and non-negative (0 disables buckets)")
    return max(1, int(megabytes * 1024 * 1024)) if megabytes else 0


def _scratch(reference: torch.Tensor, capacity: int, pool: str) -> torch.Tensor:
    key = (pool, reference.device, reference.dtype)
    cached = _BUCKET_BUFFERS.get(key)
    if cached is None or cached.numel() < capacity or (pool == "gradient" and cached.numel() != capacity):
        # Drop an old-size bucket before allocating its replacement, including
        # when switching 64 -> 32 MiB in an in-process benchmark.
        _BUCKET_BUFFERS.pop(key, None)
        cached = None
        cached = torch.empty(capacity, device=reference.device, dtype=reference.dtype)
        _BUCKET_BUFFERS[key] = cached
    return cached


@torch.no_grad()
def _bucketed_sync(tensors: Iterable[torch.Tensor], *, max_bytes: int, pool: str,
                   scale: float | None = None, src: int | None = None) -> None:
    # Insertion order is deterministic when ranks have the same tensor/None
    # layout, as required by the existing per-parameter collectives as well.
    groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for tensor in tensors:
        groups.setdefault((tensor.device, tensor.dtype), []).append(tensor)

    def collective(tensor):
        if src is None:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        else:
            dist.broadcast(tensor, src=src)
        if scale is not None:
            tensor.mul_(scale)

    for group in groups.values():
        capacity = max(1, max_bytes // group[0].element_size())
        bucket: list[torch.Tensor] = []
        used = 0

        def flush():
            if not bucket:
                return
            if len(bucket) == 1:
                collective(bucket[0])
                return
            scratch_capacity = capacity if pool == "gradient" else used
            packed = _scratch(bucket[0], scratch_capacity, pool)[:used]
            offset = 0
            for tensor in bucket:
                count = tensor.numel()
                # Shape the destination instead of flattening a non-contiguous
                # gradient, which could allocate another full tensor copy.
                packed[offset:offset + count].view_as(tensor).copy_(tensor)
                offset += count
            collective(packed)
            offset = 0
            for tensor in bucket:
                count = tensor.numel()
                tensor.copy_(packed[offset:offset + count].view_as(tensor))
                offset += count

        for tensor in group:
            count = tensor.numel()
            if count == 0:
                continue
            if tensor.layout != torch.strided or count > capacity:
                flush()
                bucket, used = [], 0
                # An already-large tensor needs no packing; never allocate an
                # oversized staging buffer or change sparse collective behavior.
                collective(tensor)
                continue
            if used + count > capacity:
                flush()
                bucket, used = [], 0
            bucket.append(tensor)
            used += count
        flush()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else int(os.environ.get("WORLD_SIZE", "1"))


def barrier() -> None:
    if is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


@torch.compiler.disable
def average_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    if not is_distributed():
        return
    precision = os.environ.get("UFO_GRAD_COMM_PRECISION", "fp32")
    if precision not in ("fp32", "scaled_fp16"):
        raise ValueError("UFO_GRAD_COMM_PRECISION must be fp32 or scaled_fp16")
    if precision == "scaled_fp16":
        from .distributed_precision import scaled_fp16_average_gradients

        parameters = tuple(parameters)
        if scaled_fp16_average_gradients(parameters):
            return
    scale = 1.0 / float(dist.get_world_size())
    max_bytes = _gradient_bucket_bytes()
    if max_bytes:
        _bucketed_sync((param.grad for param in parameters if param.grad is not None),
                       max_bytes=max_bytes, pool="gradient", scale=scale)
        return
    for param in parameters:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.mul_(scale)


@torch.compiler.disable
def broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


@torch.compiler.disable
def sync_floating_buffers(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    scale = 1.0 / float(dist.get_world_size())
    if os.environ.get("UFO_COALESCE_SYNC", "1") != "0":
        floating, integer = [], []
        for buffer in module.buffers():
            if buffer.dtype.is_floating_point or buffer.dtype.is_complex:
                floating.append(buffer.data)
            else:
                integer.append(buffer.data)
        _bucketed_sync(floating, max_bytes=_SYNC_BUCKET_BYTES, pool="auxiliary", scale=scale)
        _bucketed_sync(integer, max_bytes=_SYNC_BUCKET_BYTES, pool="auxiliary", src=src)
        return
    for buffer in module.buffers():
        if buffer.dtype.is_floating_point or buffer.dtype.is_complex:
            dist.all_reduce(buffer.data, op=dist.ReduceOp.SUM)
            buffer.data.mul_(scale)
        else:
            dist.broadcast(buffer.data, src=src)


@torch.compiler.disable
def broadcast_optimizer_state(optimizer: torch.optim.Optimizer, src: int = 0) -> None:
    if not is_distributed():
        return
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                dist.broadcast(value.data, src=src)


def agent_optimizers(agent: Any) -> list[torch.optim.Optimizer]:
    names = [
        "actor_optimizer",
        "backward_optimizer",
        "forward_optimizer",
        "critic_optimizer",
        "discriminator_optimizer",
        "aux_critic_optimizer",
    ]
    return [getattr(agent, name) for name in names if hasattr(agent, name)]


def broadcast_agent_state(agent: Any, src: int = 0) -> None:
    if not is_distributed():
        return
    broadcast_module_state(agent._model, src=src)
    for optimizer in agent_optimizers(agent):
        broadcast_optimizer_state(optimizer, src=src)


@torch.compiler.disable
def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_distributed():
        return value
    objects = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


@torch.compiler.disable
def module_sync_report(module: torch.nn.Module, src: int = 0) -> dict[str, Any]:
    if not is_distributed():
        return {"world_size": 1, "rank": 0, "max_abs_diff_from_rank0": 0.0}

    tensors = list(module.parameters()) + [
        buffer for buffer in module.buffers() if buffer.dtype.is_floating_point or buffer.dtype.is_complex
    ]
    if tensors:
        device = tensors[0].device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_summary = torch.zeros(3, dtype=torch.float64, device=device)
    max_abs_diff = torch.zeros((), dtype=torch.float32, device=device)

    for tensor in tensors:
        detached = tensor.detach()
        local_summary[0] += detached.double().sum()
        local_summary[1] += detached.double().square().sum()
        local_summary[2] += detached.numel()
        reference = detached.clone()
        dist.broadcast(reference, src=src)
        if detached.numel() > 0:
            diff = (detached - reference).abs().max().float()
            max_abs_diff = torch.maximum(max_abs_diff, diff)

    dist.all_reduce(max_abs_diff, op=dist.ReduceOp.MAX)
    gathered = [torch.empty_like(local_summary) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_summary)
    summaries = torch.stack(gathered).cpu()
    return {
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "max_abs_diff_from_rank0": float(max_abs_diff.cpu().item()),
        "rank_param_buffer_sum": [float(x) for x in summaries[:, 0].tolist()],
        "rank_param_buffer_sqsum": [float(x) for x in summaries[:, 1].tolist()],
        "rank_param_buffer_numel": [int(x) for x in summaries[:, 2].tolist()],
    }


@torch.compiler.disable
def average_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not is_distributed():
        return dict(metrics)
    reduced: dict[str, Any] = {}
    scale = 1.0 / float(dist.get_world_size())
    coalesce = os.environ.get("UFO_COALESCE_SYNC", "1") != "0"
    for key, value in metrics.items():
        if torch.is_tensor(value):
            tensor = value.detach().clone()
            if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
                tensor = tensor.float()
            if coalesce:
                # Return independent clones, never views of reusable scratch:
                # the caller retains these values across subsequent updates.
                reduced[key] = tensor
            else:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                reduced[key] = tensor * scale
        else:
            reduced[key] = value
    if coalesce:
        _bucketed_sync((value for value in reduced.values() if torch.is_tensor(value)),
                       max_bytes=_SYNC_BUCKET_BYTES, pool="auxiliary", scale=scale)
    return reduced
