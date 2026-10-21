"""Build measured Adam motor groups with one configured command-delay range."""

from numbers import Integral

from mjlab.actuator import BuiltinPdActuatorCfg

from .torque_speed import TorqueSpeedEnvelopeActuatorCfg


def build_grouped_actuators(groups, dof_names, *, enable_delay, delay_step_range, physics_dt):
    """Use domain_rand's delay range; group parameters only describe the motors."""
    if (
        len(delay_step_range) != 2
        or any(not isinstance(value, Integral) or isinstance(value, bool) for value in delay_step_range)
        or not 0 <= delay_step_range[0] <= delay_step_range[1]
    ):
        raise ValueError("delay_step_range must contain two ordered, nonnegative integer physics steps")
    min_lag, max_lag = delay_step_range if enable_delay else (0, 0)
    if max_lag and abs(physics_dt - 0.005) > 1e-12:
        raise ValueError("Adam command delay requires a 5 ms physics step")

    result = []
    covered = []
    for group in groups:
        names = tuple(group["joints"])
        covered.extend(names)
        params = dict(group["parameters"])
        # Historical profiles contain these fields. The explicit domain-rand
        # range is now authoritative, including when the group omits them.
        params.pop("delay_min_lag", None)
        params.pop("delay_max_lag", None)
        params["delay_min_lag"] = int(min_lag)
        params["delay_max_lag"] = int(max_lag)
        for key, value in list(params.items()):
            if isinstance(value, list):
                params[key] = tuple(value)
        model = group["model"]
        if model == "torque_speed_envelope":
            cls = TorqueSpeedEnvelopeActuatorCfg
        elif model == "builtin_pd":
            cls = BuiltinPdActuatorCfg
        else:
            raise ValueError(f"Unsupported actuator group model: {model}")
        result.append(cls(target_names_expr=names, **params))
    if len(covered) != len(dof_names) or set(covered) != set(dof_names):
        raise ValueError("Actuator groups must cover each robot joint exactly once")
    return result
