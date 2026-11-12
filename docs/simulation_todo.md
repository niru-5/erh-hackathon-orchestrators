# Simulation TODO: task v1 (single pick-and-place)

Working checklist for the first goal in [simulation.md](simulation.md): **build the SO-101 + table + board
world in MuJoCo, check it against the real arm, add real pieces and calibrated cameras, then pick and place
one piece in sim and compare with the real world.** We go through it one step at a time.

Legend: **[you]** = needs the real rig / your input, **[me]** = I implement in the repo, **[both]** = needs both.
Each step has an **output** (what lands in the repo) and a **done when** (how we know it works).

## Part A: Robot and table in MuJoCo

### A1. Set up the MuJoCo environment [me] ✅ done
- [x] Create `simulation/.venv-sim`, install `mujoco`, `mink`, `numpy`, `opencv-python`, `gymnasium` (in `simulation/requirements.txt`).
- [x] Check headless rendering (`MUJOCO_GL=egl`) works on this machine.
- [x] Create the `simulation/` skeleton from the layout in simulation.md §3 (`assets/`, `config/world.yaml`, `mujoco_env/`, `tests/`).
- **Output:** `simulation/requirements.txt`, a smoke test that loads an empty scene, steps it and renders one frame.
- **Done when:** the smoke test passes on this machine, with and without a display.

### A2. Import the SO-101 URDF [me] ✅ done (open items below)
- [x] Get the SO-101 URDF + meshes from `TheRobotStudio/SO-ARM100` ("new calibration" variant, commit recorded in `simulation/assets/so101/README.md`).
- [x] MJCF: upstream ships one, used unmodified (meshes, inertias). **Still open:** simplify the collision shapes (esp. the gripper jaws, whose convex hulls may fill the gap) and filter self-collisions; do this with the gripper/piece work (B2/B5).
- [x] Joint order `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (tested). Limits are the upstream ones (pan ±110°, lift ±100°, elbow ±96.8°, wrist ±95°, roll −157..163°, gripper −10..100° in sim units); **checking them against our arm happens in A5/A6**.
- [x] Position actuators with upstream gains (kp 998, kv 2.7, force ±3.35); tuned in A6.
- [x] `view_robot.py`: prints the joint/actuator table, renders the zero pose and each joint at its limits (headless OK), `--viewer` for the interactive window (not yet tried on your display).
- **Output:** `simulation/assets/so101/so101.xml` + meshes, `simulation/mujoco_env/view_robot.py`.
- **Done when:** the arm loads without warnings, each joint moves in the expected direction, rest pose looks like the real one.

### A3. Gather real-world setup data [you]
Put files in `simulation/real_setup/` (`photos/`, `lighting/`, `measurements/`; see its README). Anything unknown, say so and we'll measure it later.
- [ ] **Table:** height from the floor, top dimensions, surface material/color, edge positions relative to the robot.
- [ ] **Robot base:** where it is mounted on the table (distance from table edges), mounting orientation, how it is clamped.
- [ ] **Board:** outer dimensions, playing-area dimensions, square size, border width, thickness, material, a1 corner location.
- [ ] **Board to robot geometry:** measured offsets from the robot base (center of the shoulder-pan axis) to the board's a1 corner, and the board's rotation relative to the robot's forward axis. Include a top-down sketch with the numbers. Mark the points you measured so I can compute them again.
- [ ] **Photos:** top-down, front, and two side views of the whole table, with a ruler or known object in frame; plus a few close-ups of the board and robot base.
- [ ] **Lighting photos:** the same view of the table under each lighting condition you will use (day, evening, lamps on or off, etc.), noting what lights are on.
- [ ] A short note on what the two cameras currently look at (`wrist` and `side` mounting).
- **Output:** the raw data + `simulation/config/world.yaml` filled with the measurements (me, from your data).

### A4. Build the table, board and robot scene [me]
- [ ] Table, board (textured 8×8, correct square size), and robot placed using the A3 numbers in `world.yaml`.
- [ ] `square_center(file, rank)` helper, a single place for board coordinates.
- [ ] Lighting roughly matched to the photos (a first pass; photoreal lighting is Isaac Sim's job).
- [ ] Render the scene from approximately the real `side` view and compare with your photo.
- **Output:** `simulation/mujoco_env/scene.xml` (generated from `world.yaml`), side-by-side comparison image.
- **Done when:** you agree the layout matches the real table: distances and angles between robot, board and table edges look right in the overlay.

### A5. Calibrate the real arms and record calibration data [you]
- [ ] Calibrate the **follower** and the **leader** with the usual lerobot procedure. Save the calibration JSON files (and the lerobot/firmware versions) in `simulation/real_setup/calibration/`.
- [ ] Record a small set of **reference poses** on the follower: rest, each joint at its two extremes, and ~10 mixed poses. For each, save joint readings and a photo of the arm from the `side` view (so we can check the sim pose against the real one).
- [ ] Record **motion logs** for tuning: a step, a slow sine and a fast sine per joint, plus a gripper open/close. Use `observation.state` and `action` at 30 fps (a short `lerobot-record` run is fine).
- [ ] Record a few **free-form reaches** across the board and a **gripper close on nothing**, for the check in A7.
- **Output:** calibration JSONs, reference-pose table + photos, motion logs.

### A6. Calibration adapter and actuator tuning [me]
- [ ] `simulation/calibration.py`: map between sim radians and lerobot degrees/% from the calibration JSON, with unit tests.
- [ ] Verify with the reference poses: the sim arm rendered from `side` overlays the real photo within a few pixels at several poses.
- [ ] Fit actuator gains (kp, kd, armature, friction loss, force limit) to the motion logs; check lag and overshoot.
- **Output:** `simulation/calibration.py`, tuned actuator parameters in `world.yaml`, overlay images and RMS error table.
- **Done when:** joint-state RMS error between sim and real replay is within a few degrees per joint, and pose overlays visually match.

### A7. Can the sim follower do what the real one does? [both]
- [ ] **[me]** Replay the recorded real `action` streams open-loop in sim and compare with the real `observation.state` (plots per joint).
- [ ] **[me]** Reach test: command the end-effector to each square's center in sim with IK, report which squares are reachable and the joint solutions.
- [ ] **[you]** Run the same reach targets on the real follower (via the leader or the same joint commands) and tell me where it succeeds or struggles. Compare with the sim's reachability map.
- [ ] Note the discrepancies (lag, sag under gravity, backlash, gripper range) and fix or record them as randomization ranges.
- **Output:** comparison plots, a reachability map, a short findings note.
- **Done when:** sim and real agree on which squares are reachable, and the replay error is small enough that we trust the sim arm for the piece work.

## Part B: Pieces, cameras and the first pick and place

### B1. Photos and measurements of the pieces [you]
- [ ] For each of the 6 piece types (and both colors if they differ): weight (grams, kitchen scale), height, base diameter, neck/stem width, the widest point, and the gripping height we expect to use.
- [ ] Photos of each type from the front, side and top, with a ruler or calibration object next to it, on a plain background.
- [ ] Note the material (plastic, wood, felt base, weighted base?) and surface finish (glossy or matte).
- [ ] For task v1, say which piece type and which two squares you want (for example a pawn from c3 to d5).
- **Output:** `simulation/real_setup/pieces/` with photos and a measurements table.

### B2. Piece meshes with real dimensions and weights [me, with your photos]
- [ ] Get a mesh per piece type: use the photos to model each one (lathe profile for the symmetric pieces, a hand-built knight), or photogrammetry or a matching public mesh scaled to your measurements. We pick the cheapest route that matches the measured height and base to about a millimeter.
- [ ] Separate **visual mesh** and **simplified collision shapes** (cylinders/capsules/convex hulls) per type.
- [ ] Set mass from the measured weight; compute inertia from the collision shapes; set the center of mass if the base is weighted.
- [ ] Add the pieces to the scene through `world.yaml` (`build_scene.py`), placed on squares.
- [ ] Contact tuning: friction on pieces, board and fingertips (start with defaults in simulation.md §5 step 4).
- **Output:** `simulation/assets/chess/` meshes, per-piece parameters in `world.yaml`.
- **Done when:** piece dimensions in sim match the real ones to ~1 mm and the grams match; the full board stands untouched for 10 s with nothing moving more than 1 mm.

### B3. Calibrate the two cameras [you]
For both `wrist` and `side`:
- [ ] Print a checkerboard (or ChArUco) target, record 20 to 40 images at **640×480**, the resolution we use, at different positions and angles, sharp, with the same focus and exposure settings as in use. Fix the camera auto-focus if possible.
- [ ] Run the calibration (OpenCV) or hand me the images and I will run it. We need **intrinsics** (focal lengths, principal point, distortion) per camera.
- [ ] **Extrinsics for `side`:** with a target or the board visible, record an image so we can solve the camera pose relative to the board (and so relative to the robot base). Don't move the camera afterwards, and mark its position on the table.
- [ ] **Extrinsics for `wrist`:** the mount pose relative to the gripper. Measure it from the mount or solve it with a hand-eye calibration (the target stays fixed, the arm moves to ~15 poses; record joint states and an image at each).
- **Output:** `simulation/real_setup/cameras/` with images, plus `intrinsics_{wrist,side}.json` and extrinsics.

### B4. Cameras in the sim [me]
- [ ] Compute intrinsics and extrinsics from B3 (script in `simulation/calibration/`), with reprojection error reported.
- [ ] Create the sim cameras with the same FOV/principal point, resolution and pose; for the real lens distortion apply the undistort or distort step to the sim images so they match.
- [ ] Side-by-side check: sim vs real frames for the same joint state and board (the corners of the board should land on the same pixels).
- **Output:** camera blocks in `scene.xml` generated from the calibration files, an overlay review image.
- **Done when:** board corners and the robot's visible links agree between sim and real to a few pixels in both cameras.

### B5. Pick and place one piece in simulation [me]
- [ ] `ChessEnv` with `reset`, `step`, observation (`state`, `wrist`, `side`), `success()`.
- [ ] IK expert for the chosen piece and squares: pre-grasp, grasp, close, lift, transit, place, release, retreat.
- [ ] Run over many seeds with small randomization on piece pose and start pose; report the success rate and failure causes.
- [ ] Record the successful episodes to a LeRobotDataset v3.0 (`_sim` suffix, `wrist` + `side`, 30 fps) and make a video.
- **Output:** `simulation/expert.py`, `simulation/env.py`, sim dataset and videos.
- **Done when:** ≥ 90 % success over 100 seeds for the chosen piece, and you have watched the videos and they look plausible.

### B6. Transfer to the real world and compare [both]
- [ ] **[me]** Export the expert's joint trajectories for the same piece and squares, in the real arm's degrees/% convention, with a speed limit and workspace guard.
- [ ] **[you]** Play them on the real follower, slowly, one piece on the board, hand on the power, and record video from both cameras and the joint log.
- [ ] **[both]** Compare sim vs real for each: did the grasp succeed, where did it differ (approach height, slip, gripper force, placement error in mm), and what does that say about friction, gripper force, calibration or camera error?
- [ ] Update `world.yaml` and the actuator/contact tuning from what we learn, and repeat.
- **Output:** comparison note with videos and the list of fixes.
- **Done when:** the same pick and place works in sim and on the real arm, and the placement error is within a fraction of a square.

## Order of work

```
A1 → A2 ─┐
A3 ──────┼→ A4 → A6 → A7 ─┐
A5 ──────┘                │
B1 → B2 ──────────────────┼→ B5 → B6
B3 → B4 ──────────────────┘
```

**Start now, in parallel:** A1 and A2 are mine, and A3, A5 and B1 are measurements and photos you can take in any order.

## Open questions

- Which piece type and squares for task v1?
- Which of the existing cameras (`scene` or `extra`) becomes `side`, and will it stay fixed on the table?
- Does the robot base stay clamped in the same spot, or can it move between sessions? (This decides whether the base pose must be re-measured each time.)
