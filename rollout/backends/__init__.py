"""Backends: how the robot side reaches a model. Both expose ``spec()``, ``predict(window)`` and ``close()``."""

from typing import Protocol

from ..contract import Plan, PolicySpec, Window


class Backend(Protocol):
    def spec(self) -> PolicySpec: ...

    def predict(self, window: Window) -> Plan: ...

    def close(self) -> None: ...


def make_backend(policy: str, **kwargs) -> Backend:
    """``http(s)://...`` -> RemoteBackend; ``local:<model>`` -> that model in this process."""
    if policy.startswith(("http://", "https://")):
        from .remote import RemoteBackend

        return RemoteBackend(policy, **kwargs)
    if policy.startswith("local:"):
        from ..models import make_model
        from .local import LocalBackend

        return LocalBackend(make_model(policy.removeprefix("local:"), **kwargs))
    raise ValueError(f"--policy must be an http(s):// URL or local:<model>, got {policy!r}")
