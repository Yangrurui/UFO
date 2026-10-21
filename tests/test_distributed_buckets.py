"""Compare bucketed collectives with the original path on two real CPU ranks."""

import copy
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from humanoidverse import distributed

_TINY_BUCKET_BYTES = 64
_TINY_BUCKET_MB = str(_TINY_BUCKET_BYTES / 2**20)


def _mode(optimized):
    os.environ["UFO_GRAD_BUCKET_MB"] = _TINY_BUCKET_MB if optimized else "0"
    os.environ["UFO_COALESCE_SYNC"] = "1" if optimized else "0"


def _assert_same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _gradient_parameters(rank):
    # Interleave dtypes; the two ranks have the same None/empty gradient layout.
    gradients = [
        torch.arange(3, dtype=torch.float32) + rank * 2,
        torch.arange(2, dtype=torch.float64) + rank * 2,
        torch.arange(5, dtype=torch.float32) + rank * 2,
        torch.empty(0, dtype=torch.float32),
        None,
        (torch.arange(12, dtype=torch.float32).reshape(3, 4) + rank * 2).t(),
        torch.arange(3, dtype=torch.float64) + rank * 2,
        torch.arange(31, dtype=torch.float32) + rank * 2,
        torch.arange(20, dtype=torch.float64) + rank * 2,
        torch.arange(2, dtype=torch.float32) + rank * 2,
    ]
    parameters = []
    for gradient in gradients:
        parameter = torch.nn.Parameter(torch.zeros(2) if gradient is None else torch.zeros_like(gradient))
        parameter.grad = gradient
        parameters.append(parameter)
    assert not parameters[5].grad.is_contiguous()
    return parameters


def _gradient_checks(rank):
    baseline = _gradient_parameters(rank)
    _mode(False)
    distributed.average_gradients(iter(baseline))
    expected = [None if p.grad is None else p.grad.clone() for p in baseline]
    other_rank = _gradient_parameters(1 - rank)
    local = _gradient_parameters(rank)
    for index, (left, right) in enumerate(zip(local, other_rank)):
        if left.grad is not None:
            _assert_same(expected[index], (left.grad + right.grad) / 2)

    distributed._BUCKET_BUFFERS.clear()
    _mode(True)
    pointers = None
    real_all_reduce = dist.all_reduce
    for repetition in range(2):
        parameters = _gradient_parameters(rank)
        original_gradients = [p.grad for p in parameters]
        oversized = {
            p.grad.data_ptr()
            for p in parameters
            if p.grad is not None and p.grad.numel() * p.grad.element_size() > _TINY_BUCKET_BYTES
        }
        calls = []

        def observed_all_reduce(tensor, *args, **kwargs):
            calls.append((tensor.data_ptr(), tensor.numel() * tensor.element_size()))
            return real_all_reduce(tensor, *args, **kwargs)

        dist.all_reduce = observed_all_reduce
        try:
            distributed.average_gradients(iter(parameters))
        finally:
            dist.all_reduce = real_all_reduce
        for parameter, original, reference in zip(parameters, original_gradients, expected):
            assert parameter.grad is original
            if reference is None:
                assert parameter.grad is None
            else:
                _assert_same(parameter.grad, reference)
        assert oversized.issubset({pointer for pointer, _ in calls})
        assert all(size <= _TINY_BUCKET_BYTES or pointer in oversized for pointer, size in calls)
        assert len(calls) < sum(p.grad is not None for p in parameters)
        cache = {key: value for key, value in distributed._BUCKET_BUFFERS.items() if key[0] == "gradient"}
        assert cache
        assert {key[2] for key in cache} == {torch.float32, torch.float64}
        for key, tensor in cache.items():
            assert key[1] == tensor.device
            assert 0 < tensor.numel() * tensor.element_size() <= _TINY_BUCKET_BYTES
        current = {key: value.data_ptr() for key, value in cache.items()}
        if repetition:
            assert current == pointers, "Repeated calls must reuse bounded gradient scratch storage"
        pointers = current


def _metrics(rank):
    metrics = {f"scalar_{i}": torch.tensor(float(i + rank), requires_grad=True) for i in range(41)}
    metrics.update(
        double=torch.tensor(0.125 + rank, dtype=torch.float64),
        integer=torch.tensor(rank, dtype=torch.int64),
        boolean=torch.tensor(bool(rank)),
        matrix=(torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank).t(),
        empty=torch.empty(0),
        label=f"rank {rank}",
        python_number=rank,
        metadata={"rank": rank},
    )
    return metrics


def _metric_checks(rank):
    metrics = _metrics(rank)
    originals = {key: value.detach().clone() for key, value in metrics.items() if torch.is_tensor(value)}
    _mode(False)
    expected = distributed.average_metrics(metrics)
    _mode(True)
    actual = distributed.average_metrics(metrics)
    assert list(actual) == list(metrics)
    for key, value in actual.items():
        if torch.is_tensor(value):
            _assert_same(value, expected[key])
            _assert_same(metrics[key], originals[key])
            assert not value.requires_grad
            assert value is not metrics[key]
        else:
            assert value is metrics[key]
    assert actual["integer"].dtype == torch.float32
    assert actual["integer"].item() == actual["boolean"].item() == 0.5
    for index in range(41):
        assert actual[f"scalar_{index}"].item() == index + 0.5
    # A later reduction must not overwrite tensors already returned to callers.
    later_metrics = _metrics(rank)
    for key, value in later_metrics.items():
        if torch.is_tensor(value) and value.is_floating_point():
            later_metrics[key] = value.detach() + 10
    later = distributed.average_metrics(later_metrics)
    scratch_storages = {value.untyped_storage().data_ptr() for value in distributed._BUCKET_BUFFERS.values()}
    for key, value in actual.items():
        if torch.is_tensor(value):
            _assert_same(value, expected[key])
            if value.numel():
                assert value.untyped_storage().data_ptr() != later[key].untyped_storage().data_ptr()
                assert value.untyped_storage().data_ptr() not in scratch_storages
    assert distributed.average_metrics({}) == {}


def _buffer_module(rank):
    module = torch.nn.Module()
    values = {
        "scalar": torch.tensor(float(rank)),
        "vector": torch.arange(3, dtype=torch.float32) + rank,
        "double_values": torch.arange(5, dtype=torch.float64) + rank,
        "matrix": (torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank).t(),
        "complex": torch.tensor([complex(rank, rank + 1)], dtype=torch.complex64),
        "empty": torch.empty(0),
        "count": torch.tensor(7 + rank, dtype=torch.int64),
        "counts": torch.tensor([11 + rank, 21 + rank], dtype=torch.int64),
        "index": torch.tensor([rank, rank + 2], dtype=torch.int32),
        "byte": torch.tensor([rank], dtype=torch.uint8),
        "boolean": torch.tensor(bool(rank)),
    }
    for name, value in values.items():
        module.register_buffer(name, value)
    return module


def _buffer_checks(rank):
    baseline = _buffer_module(rank)
    optimized = _buffer_module(rank)
    originals = dict(optimized.named_buffers())
    source = dict(_buffer_module(1).named_buffers())
    _mode(False)
    distributed.sync_floating_buffers(baseline, src=1)
    _mode(True)
    distributed.sync_floating_buffers(optimized, src=1)
    for name, expected in baseline.named_buffers():
        actual = getattr(optimized, name)
        assert actual is originals[name]
        _assert_same(actual, expected)
        if not (actual.is_floating_point() or actual.is_complex()):
            _assert_same(actual, source[name])
    assert optimized.scalar.item() == 0.5
    assert optimized.complex.item() == complex(0.5, 1.5)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 20)
        self.bias = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.double_weights = torch.nn.Parameter(torch.tensor([0.4, -0.6], dtype=torch.float64))
        self.intermittent = torch.nn.Parameter(torch.tensor([0.2]))
        self.unused = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def loss(self, rank, step):
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10 + rank / 3
        target = torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.1, 0.5]]) + step / 10
        output = inputs @ self.weight.t() + self.bias
        loss = (output - target).square().mean().double() + self.double_weights.square().sum() / 7
        if step == 1:
            loss = loss + self.intermittent.square().sum()
        return loss


def _adam_checks(rank):
    baseline = _TinyModel()
    optimized = copy.deepcopy(baseline)
    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.013, foreach=False)
    optimized_optimizer = torch.optim.Adam(optimized.parameters(), lr=0.013, foreach=False)
    for step in range(3):
        for use_buckets, model, optimizer in (
            (False, baseline, baseline_optimizer),
            (True, optimized, optimized_optimizer),
        ):
            _mode(use_buckets)
            optimizer.zero_grad(set_to_none=True)
            model.loss(rank, step).backward()
            distributed.average_gradients(model.parameters())
            # Match training's clipping outside the communication helper.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.35)
            optimizer.step()
        for reference, actual in zip(baseline.parameters(), optimized.parameters()):
            _assert_same(actual, reference)
            if reference.grad is None:
                assert actual.grad is None
            else:
                _assert_same(actual.grad, reference.grad)
            expected_state = baseline_optimizer.state.get(reference, {})
            actual_state = optimized_optimizer.state.get(actual, {})
            assert actual_state.keys() == expected_state.keys()
            for key, value in expected_state.items():
                if torch.is_tensor(value):
                    _assert_same(actual_state[key], value)
                else:
                    assert actual_state[key] == value
        assert optimized.unused.grad is None
        assert optimized.unused not in optimized_optimizer.state
        if step != 1:
            assert optimized.intermittent.grad is None
    assert optimized_optimizer.state[optimized.weight]["step"].item() == 3
    assert optimized_optimizer.state[optimized.intermittent]["step"].item() == 1


def _rank_worker(rank, rendezvous, case):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        if case == "gradients":
            _gradient_checks(rank)
        elif case == "metrics_and_buffers":
            _metric_checks(rank)
            _buffer_checks(rank)
        else:
            _adam_checks(rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo is required")
@pytest.mark.parametrize("case", ["gradients", "metrics_and_buffers", "adam"])
def test_bucketed_collectives_match_original_on_two_ranks(tmp_path, case):
    rendezvous = (tmp_path / "gloo_rendezvous").as_uri()
    mp.spawn(_rank_worker, args=(rendezvous, case), nprocs=2, join=True)
