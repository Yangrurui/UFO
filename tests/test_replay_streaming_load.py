"""Small CPU checks for replay restore without full-field staging or padding copies."""

import json

import h5py
import numpy as np
import pytest
import torch

from humanoidverse.agents.buffers import transition
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim


def assert_storage_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], dict):
            assert_storage_equal(left[key], right[key])
        else:
            assert left[key].dtype == right[key].dtype
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


@pytest.mark.parametrize("compressed", [False, True])
def test_partial_restore_streams_bounded_reads_and_zero_pads(tmp_path, monkeypatch, compressed):
    rows, capacity = 9, 12
    fields = {
        "observation-state": np.arange(rows * 4, dtype=np.float16).reshape(rows, 4),
        "observation-precise": np.arange(rows * 3, dtype=np.float64).reshape(rows, 3),
        "action": np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        "step": np.arange(rows, dtype=np.int64).reshape(rows, 1),
        "terminated": (np.arange(rows) % 2 == 0).reshape(rows, 1),
    }
    with h5py.File(tmp_path / "buffer.hdf5", "w") as hf:
        for key, data in fields.items():
            hf.create_dataset(key, data=data, compression="gzip" if compressed else None)
    (tmp_path / "config.json").write_text(json.dumps({
        "capacity": capacity, "device": "cuda:7", "_idx": rows, "_is_full": False,
    }))
    monkeypatch.setattr(transition, "_HDF5_LOAD_CHUNK_BYTES", 24)
    original_getitem = h5py.Dataset.__getitem__
    reads = []

    def bounded_read(dataset, selection):
        assert isinstance(selection, slice) and selection.start is not None and selection.stop is not None
        row_bytes = int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize
        assert (selection.stop - selection.start) * row_bytes <= max(24, row_bytes)
        reads.append((dataset.name, selection.start, selection.stop))
        return original_getitem(dataset, selection)

    def forbid_full_field_copy(*args, **kwargs):
        raise AssertionError("restore must not concatenate or stage a whole replay field")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_read)
    monkeypatch.setattr(torch, "cat", forbid_full_field_copy)
    monkeypatch.setattr(transition, "_to_torch", forbid_full_field_copy)
    loaded = transition.DictBuffer.load(tmp_path, device="cpu")
    assert loaded.device == "cpu" and len(loaded) == rows and not loaded._is_full
    assert sum(name == "/observation-precise" for name, _, _ in reads) == rows
    for key, data in fields.items():
        value = loaded.storage
        for part in key.split("-"):
            value = value[part]
        assert value.shape == (capacity, *data.shape[1:]) and value.device.type == "cpu"
        assert value.dtype == torch.from_numpy(data).dtype
        torch.testing.assert_close(value[:rows], torch.from_numpy(data), rtol=0, atol=0)
        assert torch.count_nonzero(value[rows:]).item() == 0


def test_full_ring_cursor_and_future_writes_survive_restore(tmp_path):
    source = transition.DictBuffer(capacity=4, device="cpu")
    source.extend({"action": np.arange(3, dtype=np.float32).reshape(3, 1)})
    source.extend({"action": np.arange(3, 6, dtype=np.float32).reshape(3, 1)})
    source.save(tmp_path)
    loaded = transition.DictBuffer.load(tmp_path, device="cpu")
    assert loaded._idx == source._idx == 2 and loaded._is_full and len(loaded) == 4
    assert_storage_equal(source.storage, loaded.storage)
    data = {"action": np.arange(6, 8, dtype=np.float32).reshape(2, 1)}
    source.extend(data)
    loaded.extend(data)
    assert loaded._idx == source._idx == 0 and loaded._is_full
    assert_storage_equal(source.storage, loaded.storage)
    loaded.save(tmp_path / "continued")
    restored = transition.DictBuffer.load(tmp_path / "continued", device="cpu")
    assert_storage_equal(restored.storage, loaded.storage)


@pytest.mark.parametrize("rows", [3, 7])
def test_legacy_cursor_inference_is_preserved(tmp_path, rows):
    source = transition.DictBuffer(capacity=7, device="cpu")
    source.extend({"action": np.arange(rows, dtype=np.float32).reshape(rows, 1)})
    source.save(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    del config["_idx"], config["_is_full"]
    config_path.write_text(json.dumps(config))
    with pytest.warns(UserWarning, match="without _idx"):
        loaded = transition.DictBuffer.load(tmp_path, device="cpu")
    assert len(loaded) == rows and loaded._idx == rows % 7 and loaded._is_full == (rows == 7)
    assert_storage_equal(source.storage, loaded.storage)


def test_multidimensional_trajectory_can_sample_and_continue_after_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    source = TrajectoryDictBufferMultiDim(
        capacity=8, device="cpu", n_dim=2, seq_length=2, end_key="truncated",
        output_key_t=["observation", "action"], output_key_tp1=["observation", "terminated"],
    )
    source.extend({
        "observation": {"state": np.broadcast_to(np.arange(5, dtype=np.float32)[:, None, None], (5, 2, 3)).copy()},
        "action": np.zeros((5, 2, 1), dtype=np.float32),
        "terminated": np.zeros((5, 2, 1), dtype=bool),
        "truncated": np.zeros((5, 2, 1), dtype=bool),
    })
    source.save(tmp_path)
    loaded = TrajectoryDictBufferMultiDim.load(tmp_path, device="cpu")
    assert loaded._idx == 5 and not loaded._is_full and loaded.size() == 10
    assert_storage_equal(source.storage, loaded.storage)
    batch = loaded.sample(batch_size=4)
    torch.testing.assert_close(batch["next"]["observation"]["state"], batch["observation"]["state"] + 1)
    loaded.extend({
        "observation": {"state": np.full((1, 2, 3), 5, dtype=np.float32)},
        "action": np.ones((1, 2, 1), dtype=np.float32),
        "terminated": np.zeros((1, 2, 1), dtype=bool),
        "truncated": np.zeros((1, 2, 1), dtype=bool),
    })
    assert len(loaded) == 6 and loaded._idx == 6
    loaded.save(tmp_path / "continued")
    restored = TrajectoryDictBufferMultiDim.load(tmp_path / "continued", device="cpu")
    assert_storage_equal(restored.storage, loaded.storage)


@pytest.mark.parametrize("shapes,message", [
    ({"action": (5, 1)}, "larger than buffer capacity"),
    ({"action": (2, 1), "reward": (3, 1)}, "different from the first shape"),
    ({"action": (3,)}, "dimensions"),
])
def test_invalid_shapes_still_fail_without_changing_checkpoint(tmp_path, shapes, message):
    path = tmp_path / "buffer.hdf5"
    with h5py.File(path, "w") as hf:
        for key, shape in shapes.items():
            hf.create_dataset(key, shape=shape, dtype=np.float32)
    before = path.read_bytes()
    with pytest.raises(AssertionError, match=message):
        transition.DictBuffer(capacity=4, device="cpu").load_hdf5(path)
    assert path.read_bytes() == before
