"""Models turn a Window into a Plan. They run where the GPU is: in-process or behind rollout.server."""

from typing import Protocol

from ..contract import Plan, PolicySpec, Window


class Model(Protocol):
    def spec(self) -> PolicySpec: ...

    def predict(self, window: Window) -> Plan: ...


MODELS = ("flux", "act", "hold")


def make_model(name: str, **kwargs) -> Model:
    """``flux`` (FLUX 3 Action SO-101, needs a GPU), ``act`` (a lerobot ACT checkpoint, CPU is fine) or ``hold``
    (mock that keeps the arm still)."""
    if name == "flux":
        from .flux import FluxModel  # imports torch and flux_action only when needed

        return FluxModel(**kwargs)
    if name == "act":
        from .act import ACTModel

        return ACTModel(**kwargs)
    if name == "hold":
        from .hold import HoldModel

        return HoldModel(delay_s=kwargs.get("delay_s", 0.0))
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
