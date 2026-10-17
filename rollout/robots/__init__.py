"""Robots: where observations come from and where actions go.

Each robot has ``realtime``, ``connect()``, ``start_episode(i)``, ``observe() -> Observation | None``
(None = no more data), ``act(action) -> sent action``, ``summary() -> dict`` and ``disconnect()``.
"""

from typing import Protocol

import numpy as np

from ..contract import Observation


class Robot(Protocol):
    realtime: bool

    def connect(self) -> None: ...

    def start_episode(self, index: int) -> None: ...

    def observe(self) -> Observation | None: ...

    def act(self, action: np.ndarray) -> np.ndarray: ...

    def summary(self) -> dict: ...

    def disconnect(self) -> None: ...
