from enum import Enum


class H5Format(str, Enum):
    """Feature H5 schema to load."""

    auto = "auto"
    trident = "trident"
    atlaspatch = "atlaspatch"


class Schedule(str, Enum):
    """Learning-rate or momentum schedule."""

    linear = "linear"
    cosine = "cosine"


class WDSchedule(str, Enum):
    """Weight-decay schedule (includes *none* to disable)."""

    linear = "linear"
    cosine = "cosine"
    none = "none"


class OptimizerChoice(str, Enum):
    """Optimizer algorithm."""

    adamw = "adamw"
    adam = "adam"
    sgd = "sgd"


class Backend(str, Enum):
    """Distributed training backend."""

    nccl = "nccl"
    gloo = "gloo"


class NormType(str, Enum):
    """Normalization type for projection head layers."""

    none = "none"
    ln = "ln"


class HeadType(str, Enum):
    """Task-head architecture."""

    linear = "linear"
    mlp = "mlp"


class TokenCapSampling(str, Enum):
    """Sampling strategy when a slide exceeds the valid-token cap."""

    deterministic = "deterministic"
    random_stratified = "random_stratified"


def enum_val(v: object) -> str:
    """Return the plain string value of a CLI enum, or ``str(v)`` for non-enums."""
    return v.value if isinstance(v, Enum) else str(v)
