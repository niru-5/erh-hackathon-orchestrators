# Scene (A4): table, board, SO-101, lights, pieces, cameras

`config/world.yaml` -> `mujoco_env/build_scene.py` -> `mujoco_env/generated/scene.xml` (git-ignored, rebuilt on every `load_model`).

## Run it
```
simulation/.venv-sim/bin/python simulation/mujoco_env/view_scene.py --layout start --viewer
simulation/.venv-sim/bin/python simulation/mujoco_env/view_scene.py --layout start --render simulation/out/sheet.png --camera sheet   # top, front, 3/4, side, wrist, free
simulation/.venv-sim/bin/python simulation/mujoco_env/view_scene.py --layout single --square e4 --piece pawn --render out/x.png --camera top
```
Cameras: free `top|front|three_quarter|free` ("front" = behind the robot looking +y), model cameras `side|wrist` (cameras.py, 1280x720; **side orientation and wrist pose are PROVISIONAL** per the cameras agent). Python: `load_model('start'|'single'|'empty'|{"e4":"P","e5":"p"}, single_square=..., single_type=...)`, then `reset_rest(model)` (keyframe `rest`, ctrl = qpos).

## How it is built
- Frame: layout frame (z=0 table top). `world.square_center(file, rank)` is the only place squares are defined: a1 centre (111.7375, 207.1375) mm, h8 (238.2625, 333.6625) mm, ranks along +x, files along +y.
- Table: 420x600 mm, 18 mm slab below z=0 (collision) with a thin textured visual top (procedural dark woodgrain along x, specular but no reflectance); visual floor plane 750 mm down.
- Board: collision box 180x185x22.1 mm + 0.2 mm textured skin (`assets/board/board_top.png`, generated, 6 px/mm: border, inlay line, 8x8 light/dark wood squares with a1 dark, 1 mm dark seam across the middle at y=267.5, no real groove).
- Robot: copy of so101.xml parsed in memory (upstream untouched), mesh paths made absolute, base at (175, 0) mm yaw +90 deg (the arm extends along base +x at zero pose, so +90 deg points it to +y), origin z = 2.4 mm (base mesh sits 2.4 mm below its origin). Wrist camera/mount inserted into the `gripper` body. Rest pose in world.yaml (`rest_qpos`), actuators hold it.
- Collision: upstream collision geoms (group 3) collide; the base has no collision, so no clamp/base-vs-table contact.
- Lights: soft headlight fill, overhead spot, window-like directional light from behind-left, lamp spot on the right; all in world.yaml.
- Physics: 2 ms, elliptic cone, impratio 10.

## Reach (crude FK sampling, test_reach_envelope)
Pan axis is 38.8 mm ahead of the base origin: at (175, 38.8) mm. Squares are at 169..302 mm from it. A near-vertical tool (approach axis within ~25 deg of straight down) with the gripper frame at 42 mm height can reach radii 43..335 mm, so every square is inside the envelope (margin 33 mm at h8/h1-far squares, ~126 mm at the near end). Pan angle (positive = toward -x): a1 +20.6 deg (r 180 mm), h8 -12.1 deg (r 302 mm), a8 +20.6/-... within +-25 deg overall. The upstream shoulder_pan range is +-110 deg. Orientation limits at the extremes still need the real IK (A7).

## Assumptions (unconfirmed)
1. Robot x = 175 mm (aligned with the board centre) and base origin at the y=0 edge; real pan-axis offset and clamp/plate height unknown.
2. Board playing-area margins (robot side 23.1 mm, far 17.3 mm, x 17.7 mm) are derived, not measured.
3. a1 dark; real board's colour parity vs the photo not verified. Seam is only painted.
4. Table slab thickness 18 mm; colours sampled from photos taken under a warm lamp.
5. Light positions/intensities are guesses until lighting photos exist.

## Top-down photo vs the sim (approximate, perspective not removed)
Measured from `photos/top_down/IMG_20261112_205042055.jpg` (table 1.44 px/mm in x, 1.5 in y): board outer edges appear at x ~ 50..238 mm, y ~ 200..375 mm, against layout 85..265 and 175..360, i.e. the board looks ~25-35 mm further toward -x and ~15-25 mm further in +y than the layout labels (phone not exactly overhead; a calibrated top view is needed to trust this). Hinge seam runs along x in both; clasp faces the robot in both; the robot base looks left of the board centre in the photo rather than aligned to it.

## Still needed
Real robot base position/orientation and plate height, board outer-edge-to-first-square on the robot side, measured board position on the table (corner distances), lighting photos (lamp on/off, window), side camera pose from calibration, wrist camera mount alignment.
