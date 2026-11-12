# Camera calibration: methods and what the first data says

Code: `simulation/tools/` (`aprilgrid.py` detection + quality, `calib_methods.py` the three solvers,
`calibrate.py` pipeline/CLI, `calib_viz.py` plots, `capture_calib.py` the UI). Tests with ground truth:
`simulation/tests/test_calibration_synthetic.py`.

## The target

Your chart is an **AprilGrid** (what Kalibr uses): 12 x 9 tags, AprilTag family 36h11, ids row-major.
Measured: tag ~16 mm, gap ~5 mm (`real_setup/cameras/board.yaml`). Verified against the images: a distortion-free
homography fits the detected tag corners to ~3 px median with the 12-column layout (next best layout: 22 px), and
no id >= 108 is ever seen, so 12 x 9 is right. The board frame is the chart rotated 180 deg (id 0 top-right).

Geometry matters: with the earlier 16.1 / 4.8 mm the RMS was 3.4 px, with 16 / 5 it is 3.2 px, and the optimum
seems to sit at an even larger pitch. A caliper measurement of the printed pitch (e.g. across all 12 tags) is
worth doing; the board dimensions go straight into the focal length / distortion.

## The three methods

| | what it is | model fitted | points used |
|---|---|---|---|
| `opencv:*` | `cv2.calibrateCamera` (Zhang closed-form init, then Levenberg-Marquardt) | k1 k2 p1 p2 k3 (5), 4 with k3 fixed, or 8 rational | all tag corners (4 per tag) |
| `kalibr:radtan4` | **Kalibr-style re-implementation** (see below): same target and `pinhole-radtan` model as Kalibr, closed-form focal init, batch bundle adjustment of intrinsics + all board poses, 3-sigma corner outlier pass | fx fy cx cy, k1 k2 r1 r2 (4) | all tag corners |
| `mrcal:OPENCV5` | the real **mrcal** (`mrcal-calibrate-cameras`), `LENSMODEL_OPENCV5`, board warp off | same 5 terms as OpenCV | gap-intersection points only (each = mean of the diagonal tag-corner pairs around it), because mrcal wants a regular grid |

**Honest caveat on Kalibr.** The real Kalibr is a ROS1 package and only ships through docker images. Running a
community docker image was blocked in this session, so I did not run it. `kalibr:radtan4` is my
re-implementation of its calibration problem (same target, same camera model, same objective); it will land
close to Kalibr but it is not Kalibr (Kalibr uses its own AprilTag detector, initialisation and view selection).
To run the real thing: `python simulation/tools/calibrate.py <role> --export-kalibr /tmp/kalibr_in`, then
`KALIBR_IMAGE=<an image you trust, e.g. built from ethz-asl/kalibr's Dockerfile_ros1_20_04> bash /tmp/kalibr_in/run_kalibr.sh`.

Every result is converted to OpenCV convention and re-scored by one `evaluate()` (poses re-fitted with solvePnP
on the tag corners), so the "common RMS" column is comparable across methods, including mrcal, which fits
different points.

mrcal's CLI requires more than `2 * min(grid)` = 16 valid points per board; partial views give 6-13 gap-intersection
points, so it is launched through a 10-line runner that lowers that per-board minimum to 8. The solver itself is
untouched.

## How the methods compare on ground truth (synthetic AprilGrid views)

Known camera: fx 900, fy 905, cx 640, cy 360, k1 -0.25, k2 0.08. Errors vs truth:

| views, tags visible, noise | method | dfx | dcx | dcy | common RMS |
|---|---|---|---|---|---|
| 25, 50%, 0.3 px | opencv:5 | -0.7 | +0.4 | -0.3 | 0.42 |
| | opencv:4 | -0.7 | +0.4 | -0.3 | 0.42 |
| | kalibr-style | -0.6 | +0.4 | -0.3 | 0.42 |
| | mrcal OPENCV5 | -0.5 | +1.1 | +1.2 | 0.42 |
| 10, 25%, 0.3 px | opencv:5 | -0.1 | +0.3 | +1.5 | 0.42 |
| | kalibr-style | -0.1 | +0.4 | +1.6 | 0.42 |
| | mrcal OPENCV5 | +2.8 | +3.6 | -2.5 | 0.42 |
| 25, 50%, 1.0 px | opencv:5 | -2.5 | +1.3 | -1.2 | 1.39 |
| | kalibr-style | -2.2 | +1.1 | -1.2 | 1.39 |
| | mrcal OPENCV5 | -1.6 | +3.2 | +3.9 | 1.40 |

With good data all three recover the camera to a few pixels. OpenCV and the Kalibr-style solver are
nearly identical (they minimise the same reprojection error over the same points; differences come from the
4-term vs 5-term model and the optimiser). mrcal is a little looser here, mostly because it only sees ~1/4 of the
points (gap intersections instead of tag corners); its strengths (splined models, calibration-uncertainty
analysis, outlier handling) need a cleaner, fuller dataset than the partial-board views simulated here.

## What your first captured images say (HD Pro Webcam C920, 36 images)

Quality gate (>= 10 tags, sharp, >= 12 px tags): **10 of 36 pass**. The USB2.0_CAM1 images are all BAD
(never more than 3 tags; the one frame I looked at is dark and motion-blurred). No reprojection outliers were found among
the 10 survivors, but those 10 are all similar views:

| method | images | common RMS px | fx | fy | cx | cy |
|---|---|---|---|---|---|---|
| opencv:5 | 10 | 3.28 | 970 | 999 | 780 | 323 |
| opencv:4 | 10 | 3.23 | 710 | 720 | 786 | 505 |
| kalibr-style | 10 | 3.23 | 778 | 768 | 883 | 512 |
| mrcal OPENCV5 | 8 | 4.77 | 1025 | 1038 | 1029 | 68 |

Same RMS (~3.2 px), but fx anywhere from 710 to 1025 and the principal point scattered over 250 px: the data
does not determine the camera. The plots show why: corners cover only 19 of 40 image cells (bottom centre),
and every view has the board at a similar pose. mrcal throws out 87 % of its points, a sign that the
observations are inconsistent (mostly blur/flexing). A C920 at 1280x720 should have fx near 900-1000 and a
principal point near (640, 360); treat the numbers above as "not yet calibrated".

## How to get a calibration worth saving

1. **Lock focus and exposure.** The C920 autofocuses, and focus changes the focal length:
   `v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0 -c focus_absolute=<value>`, and use a fixed exposure.
2. **Hold the board still** for each shot (use the 2 s interval: move, then wait for the countdown); motion blur is the
   main reason tags fail to decode.
3. **Keep the board flat**: tape the print to a rigid board. Hand-held paper bends, which shows up as ~2-3 px of residual.
4. **Vary the pose**: tilt +-30 deg about both axes, move near/far, and push the board into all corners of the image so
   the Coverage plot fills; aim for 20+ GOOD images (the "keep going until N GOOD" option helps).
5. Re-measure the printed tag size and gap with a caliper and update `board.yaml`.
