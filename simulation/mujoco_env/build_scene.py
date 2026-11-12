"""world.yaml + assets -> complete scene MJCF + MjModel.

  from build_scene import load_model, build_scene_xml
  model = load_model("start")          # or "single" / "empty" / {"e4": "P", "e5": "p"} / {"e4": ("white", "pawn")}
  data = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, data, 0)   # keyframe 'rest'

The SO-101 comes from assets/so101/so101.xml (upstream, untouched): a patched copy is placed at the base pose,
mesh paths made absolute, wrist camera/mount inserted in the 'gripper' body. Pieces and cameras come from
pieces.py / cameras.py (sibling modules by other agents); simple fallbacks are used when they are missing.
"""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from world import FILES, ROOT, board_top_z, load_world, mm, parse_square, square_center  # noqa: E402

ROBOT_XML = ROOT / "assets" / "so101" / "so101.xml"
BOARD_DIR = ROOT / "assets" / "board"
GEN_DIR = Path(__file__).resolve().parent / "generated"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BACK_RANK = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
LETTER = {"p": "pawn", "r": "rook", "n": "knight", "b": "bishop", "q": "queen", "k": "king"}

try:
    import pieces as _pieces
except Exception:  # pieces agent's module not there yet
    _pieces = None
try:
    import cameras as _cameras
except Exception:
    _cameras = None


# ---------------------------------------------------------------- textures
def _rgb(c):
    return np.array(c, dtype=np.float32)


def make_board_texture(w: dict, path: Path) -> Path:
    """Top texture of the outer board box. Image col -> +x (left to right), image row -> -y (top row = far side, y max)
    before MuJoCo's own v flip; the orientation is verified by test_scene / renders."""
    import cv2

    b = w["board"]
    k = b["texture_px_per_mm"]
    x0, x1 = b["outer_x_mm"]
    y0, y1 = b["outer_y_mm"]
    W, H = int(round((x1 - x0) * k)), int(round((y1 - y0) * k))
    xs = x0 + (np.arange(W) + 0.5) / k  # mm per column
    ys = y1 - (np.arange(H) + 0.5) / k  # row 0 = y max
    X, Y = np.meshgrid(xs, ys)
    img = np.empty((H, W, 3), np.float32)
    img[:] = _rgb(b["border_rgb"])
    # wood grain on the border (along x)
    rng = np.random.default_rng(1)
    g = cv2.GaussianBlur(rng.normal(size=(H, W)).astype(np.float32), (0, 0), sigmaX=40 * k / 6, sigmaY=1.2)
    g = g / (np.abs(g).max() + 1e-6)
    img *= (1 + 0.12 * g)[..., None]
    px0, px1 = b["playing_x_mm"]
    py0, py1 = b["playing_y_mm"]
    ix, iy = b["inlay_mm"]
    inlay = (X >= px0 - ix) & (X <= px1 + ix) & (Y >= py0 - iy) & (Y <= py1 + iy)
    img[inlay] = _rgb(b["inlay_rgb"])
    inside = (X >= px0) & (X <= px1) & (Y >= py0) & (Y <= py1)
    sq = b["square_mm"]
    file0 = np.clip(((X - px0) // sq).astype(int), 0, 7)  # file index, along x
    rank0 = np.clip(((Y - py0) // sq).astype(int), 0, 7)  # rank-1, along y
    dark = ((rank0 + file0) % 2 == 0) == bool(b.get("a1_dark", True))
    sqcol = np.where(dark[..., None], _rgb(b["dark_square_rgb"]), _rgb(b["light_square_rgb"]))
    sqcol = sqcol * (1 + 0.10 * g)[..., None]
    img[inside] = sqcol[inside]
    sm = b["seam_y_mm"]
    seam = np.abs(Y - sm) <= b["seam_width_mm"] / 2
    img[seam] *= 0.35
    img = np.clip(img, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[:, :, ::-1])
    return path


def make_table_texture(w: dict, path: Path) -> Path:
    """Dark woodgrain laminate, grain along x (as in the top-down photo). 2 px/mm over the 420x600 top."""
    import cv2

    t = w["table"]
    W, H = int(t["width_mm"] * 2), int(t["length_mm"] * 2)
    rng = np.random.default_rng(7)
    f = np.zeros((H, W), np.float32)
    for sy, sx, a in [(2.5, 150, 1.0), (6.0, 250, 0.8), (14.0, 350, 0.6)]:
        n = cv2.GaussianBlur(rng.normal(size=(H, W)).astype(np.float32), (0, 0), sigmaX=sx, sigmaY=sy)
        f += a * n / (n.std() + 1e-6)
    f = (f - f.min()) / (f.max() - f.min())
    f = np.clip((f - 0.5) * 1.6 + 0.5, 0, 1) ** 1.2
    lo, hi = _rgb(t["color_dark_rgb"]), _rgb(t["color_light_rgb"])
    img = lo + (hi - lo) * f[..., None]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1])
    return path


# ---------------------------------------------------------------- helpers
def _f(v, nd=6):
    return " ".join(f"{float(x):.{nd}g}" for x in (v if hasattr(v, "__iter__") else [v]))


def normalize_layout(layout, w: dict, **kw) -> dict[str, tuple[str, str]]:
    """-> {'e4': ('white', 'pawn'), ...}"""
    if isinstance(layout, dict):
        out = {}
        for sq, v in layout.items():
            parse_square(sq)
            if isinstance(v, str):
                out[sq] = ("white" if v.isupper() else "black", LETTER[v.lower()])
            else:
                out[sq] = (v[0], v[1])
        return out
    if layout == "empty":
        return {}
    if layout == "single":
        s = dict(w["pieces"]["single"])
        s.update({k[len("single_"):]: v for k, v in kw.items() if k.startswith("single_")})
        return {s["square"]: (s["color"], s["type"])}
    if layout == "photo_210615":  # the 3-piece layout of real_setup/cameras/side/20261112_210615
        return {"a1": ("white", "king"), "a8": ("black", "king"), "h1": ("black", "rook")}
    if layout == "start":
        out = {}
        for fi, f in enumerate(FILES):
            out[f"{f}1"] = ("white", BACK_RANK[fi])
            out[f"{f}2"] = ("white", "pawn")
            out[f"{f}7"] = ("black", "pawn")
            out[f"{f}8"] = ("black", BACK_RANK[fi])
        return out
    raise ValueError(f"unknown layout {layout!r}")


def _fallback_piece_body(ptype, color, name, pos, yaw=0.0):
    h = {"pawn": 0.030, "rook": 0.033, "knight": 0.036, "bishop": 0.040, "queen": 0.046, "king": 0.050}[ptype]
    r = {"pawn": 0.0075, "rook": 0.0085, "knight": 0.0085, "bishop": 0.0085, "queen": 0.009, "king": 0.009}[ptype]
    rgba = "0.92 0.85 0.7 1" if color == "white" else "0.15 0.1 0.08 1"
    return (f'<body name="{name}" pos="{_f(pos)}" euler="0 0 {yaw:.6g}"><freejoint name="{name}_j"/>'
            f'<geom type="cylinder" size="{r} {h / 2}" pos="0 0 {h / 2}" rgba="{rgba}" mass="0.002" friction="0.8 0.005 0.0001"/></body>')


def _robot_parts(w: dict):
    """Parse so101.xml -> (default_xml, base body Element, asset children, actuator xml)."""
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    defaults = "".join(ET.tostring(d, encoding="unicode") for d in root.findall("default"))
    base = root.find("worldbody").find("body[@name='base']")
    assets = root.find("asset")
    teal = w.get("robot_base", {}).get("rgba")
    for mat in assets.findall("material"):  # the printed parts of the real arm are teal, not upstream yellow
        if teal and mat.get("rgba", "").startswith("1 0.82 0.12"):
            mat.set("rgba", " ".join(str(v) for v in teal))
    for m in assets.findall("mesh"):
        m.set("file", str(ROOT / "assets" / "so101" / "assets" / m.get("file")))
    act = ET.tostring(root.find("actuator"), encoding="unicode")
    return defaults, base, assets, act


def _frag(xml: str) -> list[ET.Element]:
    return list(ET.fromstring(f"<r>{xml}</r>")) if xml and xml.strip() else []


# ---------------------------------------------------------------- scene
def build_scene_xml(world: dict | None = None, layout="start", write_to: Path | None = None, **kw) -> str:
    w = world or load_world()
    lay = normalize_layout(layout, w, **kw)
    t, b, rb, ph = w["table"], w["board"], w["robot_base"], w["physics"]
    tex_table = make_table_texture(w, BOARD_DIR / "table_wood.png")
    tex_board = make_board_texture(w, BOARD_DIR / "board_top.png")

    defaults, base, rassets, ract = _robot_parts(w)
    # robot placement
    yaw = math.radians(rb["yaw_deg"])
    base.set("pos", _f([mm(rb["origin_xy_mm"][0]), mm(rb["origin_xy_mm"][1]), mm(rb["origin_z_mm"])]))
    base.set("quat", _f([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
    # wrist camera + mount into the gripper body
    wrist_xml, mount_vis, mount_assets = "", "", ""
    if _cameras:
        wrist_xml = _cameras.wrist_camera_xml()
        mount_vis = getattr(_cameras, "wrist_mount_visual_xml", lambda: "")()
        mount_assets = getattr(_cameras, "wrist_mount_assets_xml", lambda: "")()
    grip = base.find(".//body[@name='gripper']")
    for el in _frag(wrist_xml) + _frag(mount_vis):
        grip.append(el)
    for el in _frag(mount_assets):
        rassets.append(el)
    robot_xml = ET.tostring(base, encoding="unicode")
    robot_assets = "".join(ET.tostring(e, encoding="unicode") for e in rassets)

    # pieces
    if _pieces:
        piece_assets = _pieces.piece_assets_xml()
        piece_defaults = getattr(_pieces, "default_defaults_xml", lambda: "")() or ""
    else:
        piece_assets = ('<material name="piece_white" rgba="0.92 0.85 0.7 1"/>'
                        '<material name="piece_black" rgba="0.15 0.1 0.08 1"/>')
        piece_defaults = ""
    ztop = board_top_z(w)
    bodies = []
    for sq, (color, ptype) in sorted(lay.items()):
        x, y = (square_center(*parse_square(sq), w) if True else None)
        name = f"{color}_{ptype}_{sq}"
        fn = _pieces.piece_body_xml if _pieces else _fallback_piece_body
        bodies.append(fn(ptype, color, name, [x, y, ztop], 0.0))
    piece_xml = "\n    ".join(bodies)

    # lights
    lights = []
    for L in w["lights"]["items"]:
        d = L["dir"]
        n = np.array(d, float) / np.linalg.norm(d)
        di, sp = L["diffuse"], L.get("specular", 0.2)
        a = (f'name="{L["name"]}" pos="{_f(mm(L["pos_mm"]))}" dir="{_f(n)}" diffuse="{di} {di} {di}" '
             f'specular="{sp} {sp} {sp}" castshadow="{str(L.get("shadow", True)).lower()}"')
        if L["type"] == "directional":
            a += ' directional="true"'
        else:
            a += f' cutoff="{L.get("cutoff", 60)}" exponent="2"'
        lights.append(f"<light {a}/>")
    hl = w["lights"]["headlight"]

    tw, tl, tt = mm(t["width_mm"]), mm(t["length_mm"]), mm(t["thickness_mm"])
    bx0, bx1 = mm(b["outer_x_mm"])
    by0, by1 = mm(b["outer_y_mm"])
    bh = mm(b["height_mm"])
    side_cam = _cameras.side_camera_xml() if _cameras else ""
    tf, bf = _f(t["friction"]), _f(b["friction"])
    c = w["contact"]

    xml = f"""<mujoco model="chess_scene">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{ph['timestep']}" cone="{ph['cone']}" impratio="{ph['impratio']}" gravity="{_f(ph['gravity'])}" noslip_iterations="{ph['noslip_iterations']}"/>
  <visual>
    <headlight ambient="{hl['ambient']} {hl['ambient']} {hl['ambient']}" diffuse="{hl['diffuse']} {hl['diffuse']} {hl['diffuse']}" specular="0 0 0"/>
    <rgba haze="0.3 0.3 0.32 1"/>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
  </visual>
  {defaults}
  <default>
    <default class="piece_contact"><geom solref="{_f(c['solref'])}" solimp="{_f(c['solimp'])}" friction="{_f(c['piece_friction'])}"/></default>
    {piece_defaults}
  </default>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.35 0.37 0.4" rgb2="0.12 0.12 0.14" width="64" height="64"/>
    <texture name="tex_table" type="2d" file="{tex_table}"/>
    <material name="mat_table" texture="tex_table" specular="{t['specular']}" shininess="{t['shininess']}" reflectance="{t['reflectance']}"/>
    <texture name="tex_board" type="2d" file="{tex_board}"/>
    <material name="mat_board_top" texture="tex_board" specular="0.25" shininess="0.4"/>
    <material name="mat_board_side" rgba="{_f(np.array(b['side_rgb'])/255.0)} 1" specular="0.3" shininess="0.4"/>
    <material name="mat_floor" rgba="0.42 0.27 0.15 1" specular="0.05" shininess="0.1"/>
    {robot_assets}
    {piece_assets}
  </asset>
  <worldbody>
    {chr(10).join('    ' + l for l in lights)}
    <geom name="floor" type="plane" pos="{tw / 2} {tl / 2} {-mm(t['floor_depth_mm'])}" size="3 3 0.1" material="mat_floor" contype="0" conaffinity="0"/>
    <geom name="table_visual_top" type="box" pos="{tw / 2} {tl / 2} {-0.0002}" size="{tw / 2} {tl / 2} 0.0002" material="mat_table" contype="0" conaffinity="0"/>
    <geom name="table" type="box" pos="{tw / 2} {tl / 2} {-tt / 2 - 0.0004}" size="{tw / 2} {tl / 2} {tt / 2 - 0.0004}" rgba="0.2 0.12 0.08 1" friction="{tf}"/>
    <geom name="board" type="box" pos="{(bx0 + bx1) / 2} {(by0 + by1) / 2} {bh / 2}" size="{(bx1 - bx0) / 2} {(by1 - by0) / 2} {bh / 2}" material="mat_board_side" friction="{bf}"/>
    <geom name="board_top_visual" type="box" pos="{(bx0 + bx1) / 2} {(by0 + by1) / 2} {bh + 0.0001}" size="{(bx1 - bx0) / 2} {(by1 - by0) / 2} 0.0001" material="mat_board_top" contype="0" conaffinity="0"/>
    {side_cam}
    {robot_xml}
    {piece_xml}
  </worldbody>
  {ract}
</mujoco>"""
    if write_to is not None:
        Path(write_to).parent.mkdir(parents=True, exist_ok=True)
        Path(write_to).write_text(xml)
    return xml


def rest_qpos(w: dict) -> dict:
    return dict(w["robot_base"]["rest_qpos"])


def load_model(layout="start", world: dict | None = None, out_path: Path | None = GEN_DIR / "scene.xml", **kw) -> mujoco.MjModel:
    """Build, add keyframe 'rest' (arm rest pose, ctrl = qpos, pieces at their squares) and compile."""
    w = world or load_world()
    xml = build_scene_xml(w, layout, **kw)
    m0 = mujoco.MjModel.from_xml_string(xml)
    q = m0.qpos0.copy()
    ctrl = np.zeros(m0.nu)
    for name, v in rest_qpos(w).items():
        q[m0.joint(name).qposadr[0]] = v
        ctrl[m0.actuator(name).id] = v
    key = f'  <keyframe><key name="rest" qpos="{_f(q, 9)}" ctrl="{_f(ctrl, 9)}"/></keyframe>\n'
    xml = xml.replace("</mujoco>", key + "</mujoco>")
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(xml)
    return mujoco.MjModel.from_xml_string(xml)


def reset_rest(model: mujoco.MjModel) -> mujoco.MjData:
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("rest").id)
    mujoco.mj_forward(model, d)
    return d


if __name__ == "__main__":
    layout = sys.argv[1] if len(sys.argv) > 1 else "start"
    m = load_model(layout)
    print(f"wrote {GEN_DIR / 'scene.xml'}: nbody={m.nbody} ngeom={m.ngeom} nu={m.nu}")
