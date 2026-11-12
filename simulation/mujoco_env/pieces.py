"""Chess pieces for the MuJoCo sim: asset/body XML generators. All data from simulation/config/pieces.yaml.

Usage (scene builder):
    <asset> + piece_assets_xml() + </asset>
    <default> + default_defaults_xml() + </default>      (optional; geoms also carry explicit attributes)
    piece_body_xml("pawn", "white", "wp_e2", (x, y, z_board_surface), yaw_rad)

Piece local frame: origin = centre of the base bottom, z up. The knight's muzzle points toward local +x.
Collision geoms use default contype/conaffinity (collide with board, fingers); visual mesh geom is group 2,
contype=conaffinity=0. Collision geoms are group 3.

Physics notes (see simulation/assets/chess/check_pieces.py for the stability test):
 * Inertia is computed from the collision primitives (uniform density scaled to the measured mass),
   NOT from the mesh. Base weighting is unknown (assumed uniform; measure to confirm).
 * friction 0.8/0.01/0.001 with condim 4 is a guess; solref 0.004/1 and solimp 0.95/0.99 give a stiff, non-bouncy
   contact that keeps 1-2 g pieces from sinking or jittering at timestep 0.002 (see check_pieces.py).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

PIECE_TYPES = ["pawn", "rook", "knight", "bishop", "queen", "king"]

_REPO = Path(__file__).resolve().parents[2]
_CFG = _REPO / "simulation" / "config" / "pieces.yaml"
_MM = 1e-3


@lru_cache(maxsize=1)
def _cfg():
    with open(_CFG) as f:
        return yaml.safe_load(f)


def mesh_path(ptype):
    return str(_REPO / _cfg()["mesh_dir"] / f"{ptype}.stl")


def _check(ptype):
    if ptype not in PIECE_TYPES:
        raise ValueError(f"unknown piece type {ptype!r}; expected one of {PIECE_TYPES}")


def piece_info(ptype):
    _check(ptype)
    c = _cfg()["pieces"][ptype]
    m = c["measured"]
    return {
        "type": ptype,
        "height_m": m["height_mm"] * _MM,
        "base_diameter_m": m["base_diameter_mm"] * _MM,
        "top_diameter_m": None if m["top_diameter_mm"] is None else m["top_diameter_mm"] * _MM,
        "mass_kg": c["mass_g"] * 1e-3,
        "grasp_height_m": c["grasp_height_mm"] * _MM,
        "estimated": bool(c["estimated"]),
        "notes": c["notes"],
        "mesh": mesh_path(ptype),
    }


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def piece_assets_xml():
    cfg = _cfg()
    mat = cfg["material"]
    out = []
    for t in PIECE_TYPES:
        out.append(f'<mesh name="piece_mesh_{t}" file="{mesh_path(t)}"/>')
    for col in ("white", "black"):
        out.append(f'<material name="piece_{col}" rgba="{_fmt(cfg["colors"][col])}" '
                   f'specular="{mat["specular"]}" shininess="{mat["shininess"]}"/>')
    return "\n".join(out)


def _contact_attrs():
    k = _cfg()["contact"]
    return (f'condim="{k["condim"]}" friction="{_fmt(k["friction"])}" solref="{_fmt(k["solref"])}" '
            f'solimp="{_fmt(k["solimp"])}" margin="{k["margin"]}" gap="{k["gap"]}"')


def default_defaults_xml():
    """Content for a <default> element: class 'piece' (collision) and 'piece_visual'."""
    return (f'<default class="piece"><geom {_contact_attrs()} group="3"/></default>\n'
            f'<default class="piece_visual"><geom type="mesh" contype="0" conaffinity="0" group="2" '
            f'mass="0"/></default>')


def _prim_mass_props(p):
    """(volume, centre(3), diag inertia per unit mass about own centre (3,)) in metres/axis-aligned."""
    if p["t"] == "cyl":
        r, h = p["r"] * _MM, (p["z1"] - p["z0"]) * _MM
        v = np.pi * r * r * h
        c = np.array([0, 0, (p["z0"] + p["z1"]) / 2 * _MM])
        ixx = (3 * r * r + h * h) / 12
        return v, c, np.array([ixx, ixx, r * r / 2])
    if p["t"] == "sphere":
        r = p["r"] * _MM
        i = 0.4 * r * r
        return 4 / 3 * np.pi * r ** 3, np.array([0, 0, p["zc"] * _MM]), np.array([i, i, i])
    a, b, h = (p["x1"] - p["x0"]) * _MM, (p["y1"] - p["y0"]) * _MM, (p["z1"] - p["z0"]) * _MM
    c = np.array([(p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2, (p["z0"] + p["z1"]) / 2]) * _MM
    return a * b * h, c, np.array([(b * b + h * h) / 12, (a * a + h * h) / 12, (a * a + b * b) / 12])


@lru_cache(maxsize=None)
def _inertial(ptype):
    """Returns (mass, com(3), full inertia 3x3 about com) from the collision primitives."""
    c = _cfg()["pieces"][ptype]
    M = c["mass_g"] * 1e-3
    props = [_prim_mass_props(p) for p in c["collision"]]
    vtot = sum(v for v, _, _ in props)
    masses = [M * v / vtot for v, _, _ in props]
    com = sum(m * cc for m, (_, cc, _) in zip(masses, props)) / M
    I = np.zeros((3, 3))
    for m, (_, cc, i) in zip(masses, props):
        d = cc - com
        I += m * np.diag(i) + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    return M, com, I


def piece_inertial(ptype):
    """(mass_kg, com_xyz_m, 3x3 inertia about com). Exposed for tests."""
    _check(ptype)
    return _inertial(ptype)


def _geom_xml(p, name):
    ca = _contact_attrs() + ' group="3"'
    if p["t"] == "cyl":
        return (f'<geom name="{name}" type="cylinder" size="{p["r"]*_MM:.6g} {(p["z1"]-p["z0"])/2*_MM:.6g}" '
                f'pos="0 0 {(p["z0"]+p["z1"])/2*_MM:.6g}" {ca}/>')
    if p["t"] == "sphere":
        return f'<geom name="{name}" type="sphere" size="{p["r"]*_MM:.6g}" pos="0 0 {p["zc"]*_MM:.6g}" {ca}/>'
    pos = ((p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2, (p["z0"] + p["z1"]) / 2)
    half = ((p["x1"] - p["x0"]) / 2, (p["y1"] - p["y0"]) / 2, (p["z1"] - p["z0"]) / 2)
    return (f'<geom name="{name}" type="box" size="{_fmt([h*_MM for h in half])}" '
            f'pos="{_fmt([q*_MM for q in pos])}" {ca}/>')


def piece_body_xml(ptype, color, name, pos_m, yaw_rad=0.0):
    _check(ptype)
    if color not in ("white", "black"):
        raise ValueError("color must be 'white' or 'black'")
    c = _cfg()["pieces"][ptype]
    M, com, I = _inertial(ptype)
    q = (np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2))
    lines = [f'<body name="{name}" pos="{_fmt(pos_m)}" quat="{_fmt(q)}">',
             f'  <freejoint name="{name}_joint"/>',
             f'  <inertial pos="{_fmt(com)}" mass="{M:.8g}" diaginertia="{_fmt(np.diag(I))}"/>',
             f'  <geom name="{name}_vis" type="mesh" mesh="piece_mesh_{ptype}" material="piece_{color}" '
             f'contype="0" conaffinity="0" group="2" mass="0"/>']
    for i, p in enumerate(c["collision"]):
        lines.append("  " + _geom_xml(p, f"{name}_col{i}"))
    lines.append("</body>")
    return "\n".join(lines)
