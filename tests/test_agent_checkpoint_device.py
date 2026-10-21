import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from humanoidverse.agents.fb.agent import FBAgent


class AgentCheckpointDeviceTest(unittest.TestCase):
    def test_checkpoint_load_resolves_only_unindexed_cuda(self):
        # The model must retain the type-only device used by autocast, while
        # safetensors needs an explicit rank index to avoid allocating on GPU 0.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"model": {"device": "cuda"}}))
            with (path / "init_kwargs.pkl").open("wb") as handle:
                pickle.dump({"obs_space": None, "action_dim": 1}, handle)
            for requested, expected in (("cuda", "cuda:3"), ("cuda:2", "cuda:2"), ("cpu", "cpu")):
                with self.subTest(device=requested):
                    model = torch.nn.Linear(1, 1)
                    agent = SimpleNamespace(_model=model)
                    with patch.object(FBAgent, "config_class") as config, \
                         patch("torch.load", return_value={}) as load_optimizer, \
                         patch("torch.cuda.current_device", return_value=3) as current_device, \
                         patch("safetensors.torch.load_model") as load_model:
                        config.return_value.build.return_value = agent
                        self.assertIs(FBAgent.load(path, device=requested), agent)
                        self.assertEqual(config.call_args.kwargs["model"]["device"], requested)
                        self.assertEqual(load_optimizer.call_args.kwargs["map_location"], requested)
                        self.assertEqual(load_model.call_args.kwargs["device"], expected)
                        self.assertEqual(current_device.call_count, int(requested == "cuda"))


if __name__ == "__main__":
    unittest.main()
