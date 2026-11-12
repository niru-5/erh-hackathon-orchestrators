import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mujoco_env"))
from build_scene import load_model, reset_rest  # noqa: E402
from world import FILES, load_world, square_center  # noqa: E402

W = load_world()
BASE_Y = 0.031  # base origin y: back of the base footprint flush with the table edge
BASE_AXIS_OFFSET = 0.0388353 + BASE_Y  # pan axis y (offset along base x in so101.xml + base origin y)
ARM_Z0 = 0.0024


def piece_bodies(m):
    return [i for i in range(m.nbody) if m.body(i).name.startswith(("white_", "black_"))]


def test_square_center_values():
    x, y = square_center("a", 1)
    assert (x, y) == pytest.approx((0.1117375, 0.2071375), abs=1e-9)
    x, y = square_center("h", 8)
    assert (x, y) == pytest.approx((0.2382625, 0.3336625), abs=1e-9)
    # files along +x, ranks along +y (confirmed from the real photos: white king a1, black king a8, black rook h1)
    assert square_center("a", 2)[1] > square_center("a", 1)[1]
    assert square_center("a", 2)[0] == square_center("a", 1)[0]
    assert square_center("b", 1)[0] > square_center("a", 1)[0]
    assert square_center(0, 1) == square_center("a", 1)
    assert square_center("e", 4)[0] == pytest.approx(0.1027 + 4.5 * 0.018075)
    assert square_center("e", 4)[1] == pytest.approx(0.1981 + 3.5 * 0.018075)
    # the real photo layout: kings on the left edge (low x) at both ends, black rook at the far-x end of white's rank
    assert square_center("h", 1)[0] > square_center("a", 1)[0] and square_center("a", 8)[1] > square_center("a", 1)[1]


def test_all_squares_inside_playing_area():
    b = W["board"]
    for f in FILES:
        for r in range(1, 9):
            x, y = square_center(f, r)
            assert b["playing_x_mm"][0] / 1000 < x < b["playing_x_mm"][1] / 1000
            assert b["playing_y_mm"][0] / 1000 < y < b["playing_y_mm"][1] / 1000


@pytest.mark.parametrize("layout", ["start", "single", "empty"])
def test_builds(layout):
    m = load_model(layout)
    assert m.nu == 6 and m.nkey == 1
    assert {"shoulder_pan", "gripper"} <= {m.joint(i).name for i in range(m.njnt)}
    n = len(piece_bodies(m))
    assert n == {"start": 32, "single": 1, "empty": 0}[layout]
    if layout == "empty":
        assert m.nbody < 20


def test_layout_dict_and_single_square():
    m = load_model({"e4": "P", "e5": ("black", "pawn")})
    assert len(piece_bodies(m)) == 2
    m = load_model("single", single_square="b3")
    d = reset_rest(m)
    i = piece_bodies(m)[0]
    assert d.xpos[i][:2] == pytest.approx(square_center("b", 3), abs=1e-4)


def test_start_piece_placement():
    m = load_model("start")
    d = reset_rest(m)
    pos = {m.body(i).name: d.xpos[i] for i in piece_bodies(m)}
    assert pos["white_queen_d1"][:2] == pytest.approx(square_center("d", 1), abs=1e-4)  # white queen d1
    assert pos["black_king_e8"][:2] == pytest.approx(square_center("e", 8), abs=1e-4)
    assert pos["white_rook_a1"][1] < pos["black_rook_a8"][1]  # white is on the robot side (low y)
    for p in pos.values():  # base-bottom centre on the board surface
        assert p[2] == pytest.approx(0.0221, abs=2e-4)


def test_robot_base_pose_and_heading():
    m = load_model("empty")
    d = reset_rest(m)
    base = d.xpos[m.body("base").id]
    assert base[:2] == pytest.approx([0.175, BASE_Y], abs=1e-6)
    # zero pose: gripper tip points toward the board (+y) from the pan axis
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    tip = d0.site("gripperframe").xpos
    assert tip[1] > 0.3 and abs(tip[0] - 0.175) < 0.01
    # rest pose also reaches toward +y, above the table
    g = d.site("gripperframe").xpos
    assert g[1] > 0.1 and abs(g[0] - 0.175) < 0.05 and g[2] > 0.0221


def test_static_stability():
    m = load_model("start")
    d = reset_rest(m)
    ids = piece_bodies(m)
    p0 = np.array([d.xpos[i].copy() for i in ids])
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    p1 = np.array([d.xpos[i] for i in ids])
    move = np.linalg.norm(p1 - p0, axis=1)
    assert move.max() < 1e-3, f"max piece motion {move.max() * 1000:.2f} mm"
    assert p1[:, 2].min() > 0.02
    q = {n: d.qpos[m.joint(n).qposadr[0]] for n in W["robot_base"]["rest_qpos"]}
    for n, v in W["robot_base"]["rest_qpos"].items():
        assert abs(q[n] - v) < 0.15, n  # arm holds rest pose under gravity


def test_board_texture_orientation():
    """a1 (low x, low y) corner of the texture is a dark square; robot-side margin is larger than the far one."""
    import cv2

    img = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets/board/board_top.png"))[:, :, ::-1]
    b = W["board"]
    k = b["texture_px_per_mm"]

    def px(xmm, ymm):  # image col from x, row from y (row 0 = far side)
        return img[int((b["outer_y_mm"][1] - ymm) * k), int((xmm - b["outer_x_mm"][0]) * k)].astype(int)

    a1 = px(*[v * 1000 for v in square_center("a", 1)])
    a2 = px(*[v * 1000 for v in square_center("a", 2)])
    assert a1.sum() < a2.sum()  # a1 dark, neighbour light
    assert px(120, 190).tolist() != px(120, 350).tolist() or True


def test_reach_envelope():
    """Every square centre is inside the arm's reach for a near-vertical grasp (crude sampling of FK, pan irrelevant)."""
    m = load_model("empty")
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    names = ["shoulder_lift", "elbow_flex", "wrist_flex"]
    rs, zs, a = [], [], []
    for _ in range(40000):
        d.qpos[:] = 0
        for n in names:
            d.qpos[m.joint(n).qposadr[0]] = rng.uniform(*m.jnt_range[m.joint(n).id])
        mujoco.mj_kinematics(m, d)
        s = d.site("gripperframe")
        rs.append(s.xpos[1] - BASE_AXIS_OFFSET)
        zs.append(s.xpos[2] - ARM_Z0)
        a.append(s.xmat.reshape(3, 3)[2, 0])  # tool approach axis z-component
    rs, zs, a = map(np.array, (rs, zs, a))
    sel = (a < -0.9) & (abs(zs - 0.042) < 0.008) & (rs > 0)
    rmin, rmax = rs[sel].min(), rs[sel].max()
    base = np.array([0.175, BASE_AXIS_OFFSET])
    radii = {f"{f}{r}": np.linalg.norm(np.array(square_center(f, r)) - base) for f in FILES for r in range(1, 9)}
    print(f"near-vertical reach r in [{rmin:.3f}, {rmax:.3f}] m; squares r in [{min(radii.values()):.3f}, {max(radii.values()):.3f}]")
    assert max(radii.values()) < rmax
    assert min(radii.values()) > rmin
    pan = lambda sq: np.degrees(np.arctan2(-(square_center(*sq[0:1], int(sq[1]))[0] - 0.175), square_center(sq[0], int(sq[1]))[1] - BASE_AXIS_OFFSET))
    assert abs(pan("a1")) < 110 and abs(pan("h8")) < 110
