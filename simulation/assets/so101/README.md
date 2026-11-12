# SO-101 model

Source: <https://github.com/TheRobotStudio/SO-ARM100> (Apache-2.0, see `LICENSE-SO-ARM100`), commit
`a758567c3978dfeefe282ede0500085a48fe8f78`, folder `Simulation/SO101/`.

- `so101.xml`: copy of the official `so101_new_calib.xml` (MJCF, "new calibration": each joint's zero is the middle of its range,
  which is the same idea as the lerobot calibration, but our own calibration still has to be mapped in A6). Unmodified apart from the file name.
- `so101_new_calib.urdf`: the matching URDF, kept for reference and for the later Isaac Sim import.
- `assets/`: the STL meshes the MJCF uses, plus `wrist_camera_*.stl` (the wrist camera and its mount, not used in the MJCF yet).

Notes on the upstream model (from its README): generated with onshape-to-robot, **base collision meshes removed**; actuator
gains are adapted from the Open Duck Mini project and are not a 1:1 match to the lerobot servo gains (we tune them in A6);
the gripper is a hinge joint here (-0.17 to 1.75 rad), while lerobot uses 0 (closed) to 100 (open) %.

Changes we make to the model go in `simulation/mujoco_env/` (scene builder), not in this folder, so upstream stays diff-able.
