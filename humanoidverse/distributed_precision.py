"""Opt-in scaled FP16 gradient communication with FP32 master gradients.

This changes communication precision, not the model/optimizer dtype. Calibration
compares both reductions on the same local gradients and returns the FP32 result
to the optimizer. No error feedback or checkpoint state is introduced.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

_MAX_BUCKET_BYTES = 64 * 1024 * 1024
_METRIC_TILE_ELEMENTS = 1024 * 1024
_BUFFERS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_CALL_INDEX = 0
_CALIBRATION_RECORDS: list[dict] = []


def _configured_bucket_bytes():
    value = os.environ.get("UFO_FP16_BUCKET_MB")
    if value is None:
        return _MAX_BUCKET_BYTES
    try:
        megabytes = int(value)
    except ValueError as error:
        raise ValueError("UFO_FP16_BUCKET_MB must be an integer from 1 to 64") from error
    if not 1 <= megabytes <= 64:
        raise ValueError("UFO_FP16_BUCKET_MB must be an integer from 1 to 64")
    return megabytes * 1024 * 1024


def _buffers(device, capacity):
    cached = _BUFFERS.get(device)
    if cached is None or cached[0].numel() != capacity:
        _BUFFERS.pop(device, None)
        cached = None
        cached = (
            torch.empty(capacity, dtype=torch.float32, device=device),
            torch.empty(capacity, dtype=torch.float16, device=device),
        )
        _BUFFERS[device] = cached
    return cached


def _common_scale(packed, world_size):
    minimum, maximum = torch.aminmax(packed)
    maximum = torch.maximum(minimum.abs(), maximum.abs())
    # MAX must carry a non-finite value to every rank, including NaN inputs.
    maximum = torch.nan_to_num(maximum, nan=math.inf, posinf=math.inf)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    torch._assert_async(torch.isfinite(maximum), "Non-finite gradient in scaled FP16 communication")
    limit = 32768.0 / world_size
    exponent = (math.log2(limit) - maximum.log2()).floor().clamp_(-126, 126)
    scale = torch.exp2(exponent)
    # Correct a possible log2 rounding boundary while retaining a power of two.
    scale = torch.where(maximum * scale > limit, scale * 0.5, scale)
    return torch.where(maximum == 0, torch.ones_like(scale), scale)


def _measure(reference, wire, scale, world_size, totals):
    # Diagnostics only: bounded small tiles avoid a second full FP32 bucket.
    # Double scalar accumulators remain accurate for very large/small FP32 data.
    for offset in range(0, reference.numel(), _METRIC_TILE_ELEMENTS):
        stop = offset + _METRIC_TILE_ELEMENTS
        candidate = wire[offset:stop].float().mul_(1.0 / world_size).div_(scale)
        actual = reference[offset:stop]
        torch._assert_async(torch.isfinite(candidate).all(), "Non-finite FP16 calibration result")
        torch._assert_async(torch.isfinite(actual).all(), "Non-finite FP32 calibration reference")
        candidate64, actual64 = candidate.double(), actual.double()
        difference = candidate64 - actual64
        totals[:4].add_(torch.stack((
            actual64.square().sum(), candidate64.square().sum(),
            (candidate64 * actual64).sum(), difference.square().sum(),
        )))
        totals[4].copy_(torch.maximum(totals[4], difference.abs().max()))
        totals[5].add_((candidate == 0).sum())
        totals[6].add_((actual == 0).sum())
        totals[7].add_(((candidate == 0) & (actual != 0)).sum())


def _record_calibration(totals, *, call_index, requested_calls, elements, buckets):
    if dist.get_rank() != 0:
        return
    reference_sq, candidate_sq, dot, error_sq, maximum, zeros, reference_zeros, new_zeros = totals.cpu().tolist()
    reference_norm, candidate_norm = math.sqrt(reference_sq), math.sqrt(candidate_sq)
    error_norm = math.sqrt(error_sq)
    relative = error_norm / reference_norm if reference_norm else (0.0 if not error_norm else None)
    cosine = dot / (reference_norm * candidate_norm) if reference_norm and candidate_norm else (1.0 if reference_norm == candidate_norm else 0.0)
    _CALIBRATION_RECORDS.append({
        "call_index": call_index,
        "calibration": True,
        "optimizer_uses": "fp32_reference",
        "finite": all(math.isfinite(value) for value in (reference_sq, candidate_sq, dot, error_sq, maximum)),
        "relative_l2": relative,
        "cosine_similarity": max(-1.0, min(1.0, cosine)),
        "max_abs_error": maximum,
        "reference_l2": reference_norm,
        "candidate_l2": candidate_norm,
        "zero_fraction": zeros / elements if elements else 0.0,
        "reference_zero_fraction": reference_zeros / elements if elements else 0.0,
        "new_zero_fraction": new_zeros / elements if elements else 0.0,
        "gradient_elements": elements,
        "bucket_count": buckets,
    })
    destination = os.environ.get("UFO_COMM_CALIBRATION_OUTPUT")
    if destination:
        path = Path(destination).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "precision": "scaled_fp16",
            "world_size": dist.get_world_size(),
            "requested_calibration_calls": requested_calls,
            "calibration_complete": call_index >= requested_calls,
            "calibration_optimizer_precision": "fp32_reference",
            "post_calibration_optimizer_precision": "scaled_fp16_average",
            "call_order": "average_gradients invocation order; the active FBcprAux update has five phases",
            "calls": _CALIBRATION_RECORDS,
        }, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def scaled_fp16_average_gradients(parameters) -> bool:
    """Return False untouched for unsupported layouts; otherwise average in-place.

    Scratch is at most 64 MiB FP32 plus 32 MiB FP16 per device. Individual large
    gradients are split without a whole-gradient temporary. Noncontiguous/mixed
    dtype gradients fall back as a complete call to the original FP32 reducer.
    """
    global _CALL_INDEX
    parameters = tuple(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        return False
    if any(gradient.dtype != torch.float32 or gradient.layout != torch.strided
           or not gradient.is_contiguous() or gradient.device != gradients[0].device for gradient in gradients):
        return False
    _CALL_INDEX += 1
    requested_calls = int(os.environ.get("UFO_COMM_CALIBRATION_CALLS", "0"))
    if requested_calls < 0:
        raise ValueError("UFO_COMM_CALIBRATION_CALLS must be non-negative")
    calibrate = _CALL_INDEX <= requested_calls
    elements = sum(gradient.numel() for gradient in gradients)
    if not elements:
        return True
    world_size = dist.get_world_size()
    # Keep the allocation stable across the five differently sized phases.
    capacity = max(1, _configured_bucket_bytes() // 4)
    packed_storage, wire_storage = _buffers(gradients[0].device, capacity)
    totals = torch.zeros(8, dtype=torch.float64, device=gradients[0].device) if calibrate else None
    segments, used, bucket_count = [], 0, 0

    def reduce_bucket():
        nonlocal bucket_count
        packed, wire = packed_storage[:used], wire_storage[:used]
        scale = _common_scale(packed, world_size)
        packed.mul_(scale)
        wire.copy_(packed)
        dist.all_reduce(wire, op=dist.ReduceOp.SUM)
        torch._assert_async(torch.isfinite(wire).all(), "Non-finite scaled FP16 SUM")
        if calibrate:
            # Original gradients remain untouched until both reductions finish.
            offset = 0
            for gradient, start, count in segments:
                packed[offset:offset + count].copy_(gradient.view(-1)[start:start + count])
                offset += count
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            packed.mul_(1.0 / world_size)
            _measure(packed, wire, scale, world_size, totals)
        else:
            packed.copy_(wire)
            # Divide in this order: scale * world_size can overflow FP32.
            packed.mul_(1.0 / world_size).div_(scale)
            torch._assert_async(torch.isfinite(packed).all(), "Non-finite scaled FP16 mean")
        offset = 0
        for gradient, start, count in segments:
            gradient.view(-1)[start:start + count].copy_(packed[offset:offset + count])
            offset += count
        bucket_count += 1

    for gradient in gradients:
        start = 0
        while start < gradient.numel():
            count = min(capacity - used, gradient.numel() - start)
            packed_storage[used:used + count].copy_(gradient.view(-1)[start:start + count])
            segments.append((gradient, start, count))
            start += count
            used += count
            if used == capacity:
                reduce_bucket()
                segments, used = [], 0
    if used:
        reduce_bucket()
    if calibrate:
        _record_calibration(totals, call_index=_CALL_INDEX, requested_calls=requested_calls, elements=elements, buckets=bucket_count)
    return True
