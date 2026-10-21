"""Real two-rank CPU Gloo checks for scaled FP16 gradient communication."""

import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from humanoidverse import distributed
from humanoidverse import distributed_precision as precision

_BUCKET_BYTES = 128


def _parameters(rank, magnitude=1.0, *, noncontiguous=False):
    gradients = [
        (torch.linspace(0.031, 0.91, 7) + rank * 0.137) * magnitude,
        None,
        torch.empty(0),
        (torch.linspace(-0.61, 0.97, 71) + rank * 0.213) * magnitude,
        (torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank + 0.17) * magnitude,
    ]
    if noncontiguous:
        gradients[-1] = gradients[-1].t()
    parameters = []
    for gradient in gradients:
        parameter = torch.nn.Parameter(torch.zeros(2) if gradient is None else torch.zeros_like(gradient))
        parameter.grad = gradient
        parameters.append(parameter)
    return parameters


def _relative_l2(actual, expected):
    numerator = (actual.double() - expected.double()).norm().item()
    denominator = expected.double().norm().item()
    return numerator / denominator if denominator else numerator


def _numerics(rank):
    for magnitude in (1.0, 0.0, 1e-30, 1e30):
        actual = _parameters(rank, magnitude)
        local = _parameters(rank, magnitude)
        peer = _parameters(1 - rank, magnitude)
        expected = [None if left.grad is None else (left.grad + right.grad) * 0.5 for left, right in zip(local, peer)]
        identities = [parameter.grad for parameter in actual]
        calls = []
        original_reduce = dist.all_reduce

        def observe(tensor, op=dist.ReduceOp.SUM, *args, **kwargs):
            calls.append((tensor.dtype, tensor.numel(), op))
            if tensor.dtype == torch.float16:
                assert tensor.numel() <= _BUCKET_BYTES // 4
                assert torch.isfinite(tensor).all()
                assert tensor.abs().max().item() <= 32768 / dist.get_world_size()
            return original_reduce(tensor, op, *args, **kwargs)

        dist.all_reduce = observe
        try:
            distributed.average_gradients(parameter for parameter in actual)
        finally:
            dist.all_reduce = original_reduce
        assert any(dtype == torch.float32 and op == dist.ReduceOp.MAX for dtype, _, op in calls)
        payloads = [count for dtype, count, op in calls if dtype == torch.float16 and op == dist.ReduceOp.SUM]
        assert len(payloads) >= 3, "The 71-element gradient must be split across bounded buckets"
        assert any(count < _BUCKET_BYTES // 4 for count in payloads), "Cover a partially filled bucket"
        for parameter, identity, reference in zip(actual, identities, expected):
            assert parameter.grad is identity
            if reference is None:
                assert parameter.grad is None
                continue
            assert parameter.dtype == parameter.grad.dtype == torch.float32
            assert torch.isfinite(parameter.grad).all()
            assert _relative_l2(parameter.grad, reference) < 1e-3
            if magnitude == 0:
                assert torch.count_nonzero(parameter.grad).item() == 0
            if parameter.grad.numel():
                gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gathered, parameter.grad)
                torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)


def _fallback(rank):
    for precision_mode, noncontiguous, dtype in (
        ("scaled_fp16", True, torch.float32),
        ("scaled_fp16", False, torch.float64),
        ("fp32", False, torch.float32),
    ):
        os.environ["UFO_GRAD_COMM_PRECISION"] = precision_mode
        parameters = _parameters(rank, noncontiguous=noncontiguous)
        if dtype != torch.float32:
            for parameter in parameters:
                parameter.data = parameter.data.to(dtype)
                if parameter.grad is not None:
                    parameter.grad = parameter.grad.to(dtype)
        expected = []
        for parameter in parameters:
            if parameter.grad is None:
                expected.append(None)
            else:
                reference = parameter.grad.clone()
                dist.all_reduce(reference)
                expected.append(reference * 0.5)
        dtypes = []
        original_reduce = dist.all_reduce

        def observe(tensor, *args, **kwargs):
            dtypes.append(tensor.dtype)
            return original_reduce(tensor, *args, **kwargs)

        dist.all_reduce = observe
        try:
            distributed.average_gradients(iter(parameters))
        finally:
            dist.all_reduce = original_reduce
        assert dtypes and torch.float16 not in dtypes
        for parameter, reference in zip(parameters, expected):
            if reference is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, reference, rtol=0, atol=0)


def _nonfinite(rank):
    parameters = _parameters(rank)
    if rank == 1:
        parameters[0].grad[0] = float("nan")
    identities = [parameter.grad for parameter in parameters]
    original_values = [None if gradient is None else gradient.clone() for gradient in identities]
    calls = []
    original_reduce = dist.all_reduce

    def observe(tensor, op=dist.ReduceOp.SUM, *args, **kwargs):
        calls.append((tensor.dtype, op))
        return original_reduce(tensor, op, *args, **kwargs)

    dist.all_reduce = observe
    try:
        with pytest.raises(RuntimeError, match="Non-finite gradient"):
            distributed.average_gradients(iter(parameters))
    finally:
        dist.all_reduce = original_reduce
    assert (torch.float32, dist.ReduceOp.MAX) in calls
    assert not any(dtype == torch.float16 and op == dist.ReduceOp.SUM for dtype, op in calls)
    for parameter, identity, original in zip(parameters, identities, original_values):
        assert parameter.grad is identity
        if original is not None:
            torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0, equal_nan=True)
    # Both ranks must reject the same bucket, although only rank 1 supplied NaN.
    dist.barrier()


def _model_parameters():
    return [
        torch.nn.Parameter(torch.linspace(0.12, 0.47, 7)),
        torch.nn.Parameter(torch.linspace(-0.31, 0.67, 71)),
        torch.nn.Parameter(torch.tensor([0.9])),
    ]


def _loss(parameters, rank, step):
    target0 = torch.linspace(-0.41, 0.23, 7) + rank * 0.173 + step * 0.019
    target1 = torch.linspace(0.17, 0.83, 71) - rank * 0.137 + step * 0.027
    return (parameters[0] - target0).square().sum() + (parameters[1] - target1).square().sum()


def _calibration(rank, output):
    reference = _model_parameters()
    actual = copy.deepcopy(reference)
    reference_optimizer = torch.optim.Adam(reference, lr=0.007, foreach=False)
    actual_optimizer = torch.optim.Adam(actual, lr=0.007, foreach=False)
    quantization_observed = False
    for step in range(3):
        reference_optimizer.zero_grad(set_to_none=True)
        actual_optimizer.zero_grad(set_to_none=True)
        _loss(reference, rank, step).backward()
        _loss(actual, rank, step).backward()
        os.environ["UFO_GRAD_COMM_PRECISION"] = "fp32"
        distributed.average_gradients(iter(reference))
        os.environ["UFO_GRAD_COMM_PRECISION"] = "scaled_fp16"
        distributed.average_gradients(parameter for parameter in actual)
        for candidate, expected in zip(actual, reference):
            assert candidate.dtype == torch.float32
            if expected.grad is None:
                assert candidate.grad is None
            elif step < 2:
                torch.testing.assert_close(candidate.grad, expected.grad, rtol=0, atol=0)
            else:
                quantization_observed |= not torch.equal(candidate.grad, expected.grad)
                assert _relative_l2(candidate.grad, expected.grad) < 1e-3
        reference_optimizer.step()
        actual_optimizer.step()
        for candidate, expected in zip(actual, reference):
            if step < 2:
                torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
            else:
                torch.testing.assert_close(candidate, expected, rtol=1e-4, atol=2e-5)
            actual_state = actual_optimizer.state.get(candidate, {})
            expected_state = reference_optimizer.state.get(expected, {})
            assert actual_state.keys() == expected_state.keys()
            for key, value in actual_state.items():
                assert value.dtype == torch.float32
                if key == "step":
                    assert value.item() == step + 1
                elif step < 2:
                    torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
                else:
                    assert _relative_l2(value, expected_state[key]) < 2e-3
        assert actual[2].grad is None and actual[2] not in actual_optimizer.state
    assert quantization_observed
    dist.barrier()
    report = json.loads(Path(output).read_text())
    assert report["calibration_complete"] is True
    assert [entry["call_index"] for entry in report["calls"]] == [1, 2]
    for entry in report["calls"]:
        assert entry["calibration"] is True and entry["finite"] is True
        assert 0 <= entry["relative_l2"] < 1e-3
        assert 0.99999 < entry["cosine_similarity"] <= 1.000001
        assert entry["max_abs_error"] >= 0
        assert entry["gradient_elements"] == 78 and entry["bucket_count"] >= 3
        for key in ("zero_fraction", "reference_zero_fraction", "new_zero_fraction"):
            assert 0 <= entry[key] <= 1
    assert any(entry["relative_l2"] > 0 for entry in report["calls"])


def _rank_worker(rank, rendezvous, output, case):
    torch.set_num_threads(1)
    os.environ["UFO_GRAD_COMM_PRECISION"] = "scaled_fp16"
    os.environ["UFO_GRAD_BUCKET_MB"] = "0"
    os.environ["UFO_COMM_CALIBRATION_CALLS"] = "2" if case == "calibration" else "0"
    os.environ["UFO_COMM_CALIBRATION_OUTPUT"] = output
    precision._MAX_BUCKET_BYTES = _BUCKET_BYTES
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        if case == "numerics":
            _numerics(rank)
        elif case == "fallback":
            _fallback(rank)
        elif case == "nonfinite":
            _nonfinite(rank)
        else:
            _calibration(rank, output)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo is required")
@pytest.mark.parametrize("case", ["numerics", "fallback", "calibration", "nonfinite"])
def test_scaled_fp16_on_two_real_cpu_ranks(tmp_path, case):
    rendezvous = (tmp_path / "gloo").as_uri()
    mp.spawn(_rank_worker, args=(rendezvous, str(tmp_path / "calibration.json"), case), nprocs=2, join=True)
