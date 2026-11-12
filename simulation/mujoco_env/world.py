"""World config loading and board coordinates. The ONE place where square_center is defined.

World frame = layout frame (metres): origin at the table corner on the robot's edge, +x along that edge,
+y away from the robot, z up with z=0 at the table top. Ranks 1..8 run along +x, files a..h along +y.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORLD_YAML = ROOT / "config" / "world.yaml"
FILES = "abcdefgh"

_cache: dict | None = None


def load_world(path: str | Path | None = None) -> dict:
    global _cache
    if path is None and _cache is not None:
        return _cache
    with open(path or WORLD_YAML) as f:
        w = yaml.safe_load(f)
    if path is None:
        _cache = w
    return w


def mm(v):
    """mm -> m for a scalar or list."""
    return [x / 1000.0 for x in v] if isinstance(v, (list, tuple)) else v / 1000.0


def parse_square(sq: str) -> tuple[int, int]:
    """'e4' -> (file_index 0..7, rank 1..8)."""
    sq = sq.strip().lower()
    f, r = FILES.index(sq[0]), int(sq[1])
    if not 1 <= r <= 8:
        raise ValueError(sq)
    return f, r


def square_center(file, rank, world: dict | None = None) -> tuple[float, float]:
    """(x, y) in metres of a square centre. file: 'a'..'h' or index 0..7; rank 1..8.
    x = playing_x0 + (rank - 0.5) * sq ; y = playing_y0 + (file_index + 0.5) * sq."""
    w = world or load_world()
    b = w["board"]
    fi = FILES.index(file) if isinstance(file, str) else int(file)
    if not (0 <= fi < 8 and 1 <= int(rank) <= 8):
        raise ValueError((file, rank))
    sq = b["square_mm"]
    return (mm(b["playing_x_mm"][0] + (rank - 0.5) * sq), mm(b["playing_y_mm"][0] + (fi + 0.5) * sq))


def square_name_center(name: str, world: dict | None = None) -> tuple[float, float]:
    f, r = parse_square(name)
    return square_center(f, r, world)


def board_top_z(world: dict | None = None) -> float:
    return mm((world or load_world())["board"]["height_mm"])
