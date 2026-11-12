# Simulated cameras

Code: `mujoco_env/cameras.py`. Numbers: `config/cameras_sim.yaml` (every pose number has a source and a PROVISIONAL flag).
Intrinsics are read from `real_setup/cameras/{side,wrist}/calibration.yaml`, never copied.

## API
`load_camera_config()`, `side_camera_xml()` (into `<worldbody>`), `wrist_camera_xml()` (inside body `gripper` of
`so101.xml`), `wrist_mount_assets_xml()` / `wrist_mount_visual_xml()` (return `''` unless `wrist.show_mount_visual: true`),
`render_size(name)`, `camera_matrix(name, size=None, mode='crop')`, `rescale_intrinsics(K, src, dst, mode)`,
`dist_coeffs(name)`, `apply_real_distortion(img_sim, name)` (alias `distort_from_sim`), `undistort_to_sim(img_real, name)`.

The scene must contain `<visual><global offwidth="1280" offheight="720"/></visual>`, otherwise `mujoco.Renderer`
refuses 1280x720.

## Conventions
- World = layout frame (metres, z up from the table top).
- OpenCV camera: x right, y down, z forward. MuJoCo camera: x right, y up, looks along -z.
  So `x_mj = x_cv, y_mj = -y_cv, z_mj = -z_cv`; the XML uses `xyaxes = (x_mj, y_mj)`.
  `d.cam_xmat.reshape(3,3) @ diag(1,-1,-1)` gives the OpenCV rotation (camera axes in world).
- Intrinsics: `resolution`, `focalpixel="fx fy"`, `principalpixel="W/2-cx H/2-cy"`, `sensorsize` = nominal 1 um pixels
  (MuJoCo 4.0.1; sign of principalpixel verified by rendering). `tests/test_cameras.py` renders a sphere and compares
  its centroid with `cv2.projectPoints`: agreement < 0.5 px (side and wrist).
- MuJoCo renders an ideal pinhole. Real lenses have radtan distortion (side k1=-0.010, k2=-0.092; wrist k1=+0.088).
  `apply_real_distortion` remaps a sim render to look like the real lens (output pixel p_d samples the sim image at
  `undistortPoints(p_d)`); `undistort_to_sim` goes the other way. Train/compare in one domain, not mixed.

## Status
| Item | Status |
|---|---|
| Side intrinsics (1280x720, C920 focus 20, rms 0.58 px, 35 imgs) | calibrated |
| Wrist intrinsics (1280x720, Kalibr-style radtan4, rms 1.16 px, 40 imgs) | calibrated, weaker fit |
| Side position (0.536, 0.321, 0.343) m | SOLVED by solvePnP (see below); agrees with the user's measured (0.5, 0.4, 0.36) m to within 36/79/17 mm, which confirms the x origin beyond the table edge |
| Side orientation | SOLVED (2026-11-12): cv2.solvePnP on the 7x7 inner board corners in `real_setup/cameras/side/20261112_205932/img_001.png`, 1.58 px rms, script `tools/solve_side_pose.py`. Assumes the board is at the layout position; an overlay of the sim render on the real frame matches the board and table edges (`out/side_overlay.png`), and the robot base is within ~40 px |
| Wrist pose in `gripper` frame | PROVISIONAL guess: (0, -0.045, -0.03) m looking at the grasp centre, image-up = gripper +x. Needs hand-eye calibration (TODO B3). Override via `position_m`, `look_at_m`, `up_gripper`, `roll_deg` or `rotation_cv_override` |
| Wrist mount/camera STLs | off by default: the mount STL is in mm in its own CAD frame and its placement on the gripper is unverified |

## What to capture next
1. Side pose: one 1280x720 frame of the real side camera (focus lock applied, via `tools/capture_calib.py`) with the
   empty board in view, ideally the robot base too. Then solvePnP on the playing-area corners
   (x 102.7..247.3, y 198.1..342.7 mm, z = 22.1 mm) with the calibrated K and D; set `orientation.mode: xyaxes_cv`
   and `position_m` from the result. Also confirm the 500 mm x origin.
2. Wrist pose: hand-eye (checkerboard seen from the wrist at several arm poses) or one frame with the gripper
   tips visible, to fix the roll and offset.

## Resolution mismatch: 640x480 recordings
Datasets are recorded at 640x480 (`scripts/record_so101.sh`, CAM_WIDTH/HEIGHT) but both cameras were calibrated at
1280x720. A pure rescale only holds for 16:9 modes. For 4:3, the C920/UVC camera may crop the sensor or use a
different field of view, so `camera_matrix(name, (640,480))` (centre-crop then scale; `mode='scale'` refuses
unequal aspect ratios) is an assumption, not a measurement. Recommendation: either record at 1280x720 (or 640x360),
or calibrate at exactly the recorded 640x480 mode with the focus lock, and set `render_size`/calibration to that.
Compare one real 640x480 frame with `undistort_to_sim` output to see which assumption holds.
