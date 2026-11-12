import struct
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "simulation"))
sys.path.insert(0, str(REPO / "simulation" / "assets" / "chess"))

from mujoco_env import pieces as P  # noqa: E402

# measurements.yaml (mm): height, base diameter, top diameter (None = not measured)
MEAS = {"pawn": (17.9, 12.2, 7.6), "rook": (20.2, 12.1, 10.1), "bishop": (27.1, 12.5, 9.5),
        "queen": (27.0, 12.5, 10.0), "king": (33.8, 12.7, 10.8), "knight": (None, 12.2, None)}
MASS_G = {"pawn": 1.3, "knight": 1.875, "rook": 1.75, "bishop": 1.75, "queen": 2.0, "king": 2.5}


def load_stl(path):
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"].reshape(-1, 3).astype(float)


def test_api():
    assert P.PIECE_TYPES == ["pawn", "rook", "knight", "bishop", "queen", "king"]
    assert "piece_white" in P.piece_assets_xml() and "piece_black" in P.piece_assets_xml()
    assert P.piece_info("knight")["estimated"] and not P.piece_info("pawn")["estimated"]


@pytest.mark.parametrize("t", P.PIECE_TYPES)
def test_dimensions_and_mass(t):
    info = P.piece_info(t)
    h, bd, td = MEAS[t]
    assert info["base_diameter_m"] == pytest.approx(bd * 1e-3, abs=1e-6)
    assert info["mass_kg"] == pytest.approx(MASS_G[t] * 1e-3, abs=1e-7)
    V = load_stl(info["mesh"])
    assert V[:, 2].min() == pytest.approx(0, abs=1e-6)
    assert V[:, 2].max() == pytest.approx(info["height_m"], abs=0.3e-3)
    if h is not None:
        assert V[:, 2].max() == pytest.approx(h * 1e-3, abs=0.3e-3)
    # base diameter = max radius of the lowest 1.5 mm
    low = V[V[:, 2] < 1.5e-3]
    assert 2 * np.hypot(low[:, 0], low[:, 1]).max() == pytest.approx(bd * 1e-3, abs=0.3e-3)
    if td is not None and t != "king":
        # widest radius above the base (head / crown / top)
        up = V[V[:, 2] > {"pawn": 10e-3, "rook": 16e-3}.get(t, 15e-3)]
        assert 2 * np.hypot(up[:, 0], up[:, 1]).max() == pytest.approx(td * 1e-3, abs=0.3e-3)
    if t == "king":
        up = V[(V[:, 2] > 20e-3) & (V[:, 2] < 28e-3)]
        assert 2 * np.hypot(up[:, 0], up[:, 1]).max() == pytest.approx(td * 1e-3, abs=0.3e-3)
    m, com, I = P.piece_inertial(t)
    assert m == pytest.approx(MASS_G[t] * 1e-3) and 0 < com[2] < info["height_m"]
    assert np.all(np.linalg.eigvalsh(I) > 0)


def test_compiled_mass_and_stability():
    import check_pieces
    m, d, worst = check_pieces.run_stability()
    for t in P.PIECE_TYPES:
        assert m.body_mass[m.body(f"white_{t}").id] == pytest.approx(MASS_G[t] * 1e-3, rel=1e-6)
    for n, (drift, tilt) in worst.items():
        assert drift < 1e-4, n
        assert tilt < 0.5, n
