"""Explicit runtime selection of fused or foreach FP32 Adam execution."""
from __future__ import annotations

from typing import Any

import torch

from .distributed import agent_optimizers


def configure_fused_adam(agent: Any, enabled: bool = True) -> dict[str, int]:
    """Configure the existing UFO Adam optimizers without replacing their state.

    Callers opt in explicitly; importing this module changes no defaults. The
    enabled path places step counters with their parameters for fused Adam.
    The disabled path selects foreach Adam but leaves loaded counters in place
    until the workspace broadcasts them and restores non-fused steps to CPU.

    Fused Adam's constructor additionally sets ``_step_supports_amp_scaling``.
    Set it explicitly on every invocation, including loaded fused checkpoints,
    and clear it for rollback. Model parameters, moment tensors, hooks, Adam
    hyperparameters and optimizer object identities are retained.
    """
    optimizers = list({id(optimizer): optimizer for optimizer in agent_optimizers(agent)}.values())
    report = {"optimizer_count": len(optimizers), "parameter_groups": 0,
              "initialized_parameters": 0, "step_counters_moved": 0}
    # Validate the entire agent before changing execution flags or any state.
    for optimizer in optimizers:
        if type(optimizer) is not torch.optim.Adam:
            raise TypeError("Fused Adam runtime selection requires torch.optim.Adam")
        for group in optimizer.param_groups:
            if group.get("capturable", False) or group.get("differentiable", False):
                raise ValueError("Fused Adam runtime selection requires ordinary non-capturable Adam")
            for parameter in group["params"]:
                if (parameter.dtype != torch.float32 or parameter.layout != torch.strided
                        or parameter.device.type not in ("cpu", "cuda")):
                    raise ValueError("Fused Adam runtime selection requires dense FP32 CPU/CUDA parameters")
                state = optimizer.state.get(parameter)
                if not state:
                    continue
                step = state.get("step")
                if not torch.is_tensor(step) or step.numel() != 1 or step.dtype != torch.float32:
                    raise ValueError("Fused Adam runtime selection requires scalar FP32 step counters")
                required_moments = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if group.get("amsgrad", False) else ("exp_avg", "exp_avg_sq")
                for name in required_moments:
                    moment = state.get(name)
                    if (not torch.is_tensor(moment) or moment.dtype != torch.float32
                            or moment.shape != parameter.shape or moment.device != parameter.device):
                        raise ValueError(f"Invalid loaded Adam {name}")

    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group.update(foreach=not enabled, fused=enabled, capturable=False)
            report["parameter_groups"] += 1
            for parameter in group["params"]:
                state = optimizer.state.get(parameter)
                if not state:
                    continue
                report["initialized_parameters"] += 1
                if enabled and state["step"].device != parameter.device:
                    state["step"] = state["step"].to(device=parameter.device)
                    report["step_counters_moved"] += 1
        optimizer.defaults.update(foreach=not enabled, fused=enabled, capturable=False)
        optimizer._step_supports_amp_scaling = enabled
    return report
