# Worklog: SO-101 chess rig, 2026-10-17 to 2026-10-18

This log was put together from the Claude Code session history of this project: 19 sessions in the main
checkout and its `.claude/worktrees/*`. It records what was tried, what broke, why it broke, and what fixed it.
The detailed write-ups live in the topic docs, and this page links to them:

- [data_collection.md](data_collection.md): the recording pipeline, the 30 Hz work and the recorded datasets
- [inference.md](inference.md): the rollout architecture, every real-arm run, and the run log index
- [flux_joint_convention.md](flux_joint_convention.md): the FLUX normalization analysis
- [act_inference_experiments.md](act_inference_experiments.md): the ACT runs compared

All times are local (CEST, UTC+2), the same clock as the `rollout_logs/<YYYYmmdd_HHMMSS>` folder names. "Runs"
are on the real follower arm unless marked offline. "Offline" means a teacher-forced replay of recorded
held-out episodes (usually 25 and 30 of the c3→d5 dataset).

## Contents

1. [Timeline](#timeline)
2. [Debugging lessons](#debugging-lessons)
3. [Scoreboard](#scoreboard)
4. [Open problems](#open-problems)
5. [Branches not on `rollout`](#branches-not-on-rollout)
6. [Space spend](#space-spend)

## Timeline

### 2026-10-17, 10:46: environment setup

- Cloned LeRobot at **v0.6.0** (the tag leLab pins) into `./lerobot` and installed it and leLab editable into one
  `.venv` (Python 3.12). `uv` refuses two sources for `lerobot`, so an override file points it at the local
  clone. The override dropped lerobot's nested extras, so `datasets` was missing until they were installed
  directly. The leLab frontend was built so `lelab` can serve the UI, and all 325 leLab tests pass.
- **ROS Jazzy in `PYTHONPATH`** (from the shell profile) breaks Python tooling: its pytest plugin autoloads and
  fails (`No module named 'lark'`). Run tests with `PYTHONPATH=` or with plugin autoload off.
- The laptop has **no NVIDIA GPU** (Ryzen 7 5800HS with integrated Vega) and 15 GB of RAM, so it can record and run
  ACT on the CPU, but it can't run FLUX: about 25 GB of weights and a 32 GB GPU are needed.
- Installed the `hf` CLI globally (`~/.local/bin/hf` 2.2.0); `.venv` keeps its own `hf` 1.33.0. Both read the
  same token file.

### 12:11: recording pipeline (`scripts/record_so101.sh`)

- Wrapper around the official `lerobot-record` (v3.0 format) with `--task` and `--episodes`. Arms and cameras are
  found by serial number / device path, so re-plugging doesn't swap leader and follower.
- **Calibration:** the existing leLab files (`calibration/{robots/so_follower,teleoperators/so_leader}/so101.json`)
  are reused. `scripts/check_calibration.py` confirmed that the motors' stored values match the files, so no
  recalibration prompt appears. `scripts/check_cameras.py` grabs a test frame per camera.
- **Bugs found on the first real recordings:**
  - Datasets were saved under `local/` instead of `niru-5/`, because `hf auth whoami` prints a different format in
    the user's terminal. Fixed by asking `hf` for JSON output.
  - Folder names had two timestamps, because lerobot already appends `_<YYYYmmdd_HHMMSS>`. The script's own one
    was removed.
- **Camera names** set to the dataset spec: `right_side`→`scene` (`/dev/video6`, top view), `ff`→`wrist`
  (`/dev/video2`), `left_side`→`extra` (`/dev/video4`, recorded but not used for training). `--exclude-camera`
  was added later.
- **Units:** lerobot 0.6.0 stores degrees for the 5 arm joints and 0–100 % for the gripper (from the calibration),
  not raw ticks. `action` is the leader's reported positions, raw and absolute.
- **Keyboard (Wayland):** keys only register in the terminal running the script, not the Rerun window.
  → ends the episode (saved) or skips the rest of the reset, ← discards and re-records, and Esc stops
  (keeping the episode in progress). There is no pause.

### 13:41: the 30 Hz investigation

- **Symptom:** `Record loop is running slower (7.9 Hz)` and then `0.2 Hz`, together with Rerun
  `Sender has been blocked` warnings and `Reached memory limit of 1.5 GiB`.
- **Cause:** the Rerun live viewer was receiving three raw 640×480 streams (about 27 MB/s), couldn't keep up, and
  its full queues blocked lerobot's record loop. The CPU was idle. leLab (2.2 GiB) and Chromium were also eating
  memory.
- **Important correction:** lerobot writes `timestamp = frame_index / fps`, so a dataset *always* looks like
  exactly 30 Hz. Timestamps prove nothing. The real evidence is the frame count against wall time, plus
  per-frame wall-clock timing.
- **Checked leLab's "30 Hz" dataset** (`niru-5/test_1_20261017_135757`): it uses the same lerobot record code. Its
  only relevant difference is that it never starts the Rerun viewer.
- **The single slow warnings that remained** (14.2 Hz, 25.3 Hz) were one frame each, the first frame of an
  episode, when lerobot starts the SVT-AV1 encoders (40–106 ms). That's inside lerobot and wasn't fixed.
- **Measured:** the wrist camera ran at only 26.7 fps with default OpenCV settings and 30.2 fps with
  **V4L2 + MJPG**. Motor reads take about 1 ms per arm.
- **Fixes** (`4365fc2`): V4L2 + MJPG for every camera, the viewer off by default (`--display` opts in, with
  compressed images), no spoken prompts by default, and `scripts/record_timed.py`, which times every frame against
  the real clock and prints a per-episode verdict (OK = ≥ 29.4 real Hz and no frame over 50 ms, first frame
  excluded). The report is saved to `~/.cache/huggingface/lerobot/timing/<repo_id>.json`.
- **First verified episode:** `niru-5/so101_20261017_141741`, 596 frames, 29.91 real Hz, 0 late frames. Wrist
  camera repeated frames fell from 10.6 % to 4.7 %.
- **Decoder:** lerobot's default `torchcodec` can't load on this laptop (`libavdevice.so.60` missing). Use
  `video_backend="pyav"` / `--dataset.video_backend=pyav`, or `sudo apt install ffmpeg`. Recording isn't
  affected.

### 14:38: recording usage issues

- `--push on` → `Unknown argument: on`: `--push` is a flag and takes no value.
- `--play_sounds=true` was rejected by the wrapper. Pass lerobot flags after a bare `--`; draccus keeps the
  **last** value of a repeated flag, so it overrides the wrapper's `--play_sounds=false`.
- **`--resume` failed:** lerobot 0.6.0's `resume()` refuses to run without an explicit `root`
  (`lerobot_dataset.py:829`), and the script passed none. Fixed in `53f76cc`: with `--resume` and no `--root`,
  the script now uses the default folder and errors clearly if no dataset exists there. With `--resume`, use the
  full timestamped repo id, and `--episodes` counts *new* episodes.
- Advice for the task prompt: keep the wording byte-identical across runs (`c3` vs `D5`, `6cm` vs `10 cm` are
  different tasks to a language-conditioned policy), and note that ACT ignores the prompt entirely.

### 15:31: main dataset documented

`niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741`: 31 episodes, 13,229 frames, 7 min 21 s, 444 MB, public on
the Hub (byte-identical to the local copy). Two findings that matter for training:

- **The timing report covers only episodes 21–30.** `record_timed.py` overwrote it on each resume, so the real
  rate of episodes 0–20 can't be checked any more.
- **`wrist_roll` outlier:** 98 % of frames are within −85…−65°, but episodes 0/1 reach +67°. This distorts any
  min/max normalization, and it was excluded from the normalization quantiles later.

### 15:39: rollout architecture and the GPU Space

- Built the `rollout/` package: robot (real arm or dataset replay) × backend (in-process or HTTP) × model (FLUX,
  hold mock). There is **one request per 32-action plan**, not per frame. A test against BFL's own reference
  loop gives the same actions within 1e-4. Architecture: [inference.md](inference.md).
- **Why Docker:** HF Inference Providers can't serve this model. It's tagged `robotics`, it has no provider
  mapping, and custom code can't be uploaded. So it runs on a Docker GPU Space pinned to BFL's tested stack
  (torch 2.10 + CUDA 12.8, NATTEN `0.21.6+torch2100cu128`, transformers 5.16.1).
- **Space operations bugs** (`a81cc5e`…`36ecad3`):
  - L40S had no capacity, so the Space switched to **A100-large at $2.50/h**.
  - `start` didn't unpause after a hardware switch.
  - `start` gave up on a stale `PAUSED` stage read right after the restart.
  - Deploy logs stayed empty because Python buffers stdout to files, so the output is now flushed.
- `python -m rollout.server.space start|stop|status` keeps a local spend ledger and refuses to start past the
  budget ($20, raised to **$35** on 10-18 in `a900b60`). The Space also sleeps after 15 idle minutes.
- **First FLUX numbers:** 1.07 s of inference per plan on the A100, about 1.5 s per request. The arm moves for
  about 1 s, then holds about 1.5 s.
- **Misleading first result:** base FLUX looked "389° off" on shoulder_lift. Most of that was *the evaluation
  method*: replaying a recording while feeding the model its own commands makes the history inconsistent after
  the first plan. Fixed with **teacher forcing** (`9d1f24c`, the default for `--robot dataset`).

### 16:42: first LoRA on the arm, and the table hit

- `Gawtam/flux3-chess-lora` is a lerobot-format PEFT LoRA. The server merges it into the standalone `flux_action`
  weights at load (`93f8326`), and every adapter key matched a base weight by name and shape. Offline errors
  (5–13° per joint) matched the adapter card.
- **Real arm (step 250):** the first plan was smooth. From about 6.6 s it pushed into the table, with
  shoulder_lift stuck while the goal stayed about 10° beyond it (exactly the per-tick cap). pan and wrist_flex
  went outside the demo range (the clamp was demo range ±10°). At about 12 s the servos stopped answering
  (overload protection), and even torque-off failed. **The follower needed a power cycle.**
- **Fixes** (`f5ce70e`): a stall guard, joint limits at the demo range with no margin, and a 5°/tick cap.

### 17:26: ACT runs locally on the CPU

- `--policy local:lerobot --weights <repo>` (`bd91427`) loads any lerobot chunk policy with its own config and
  normalizers. `rtrtwrw/act_chess_c3_to_d5` (31 episodes): about 0.45 s per 100-action plan, the arm moving
  about 88 % of the time. It did a full reach → grasp → carry → release → return on the first real-arm episode.
- A parallel session added `--policy local:act` on `worktree-act-rollout` (never merged; `local:lerobot` covers
  it) and ran the 31-episode checkpoints. **Step 2000** put the knight on d5 and returned home in about 11 s.
  **Step 4000** made two runs of empty grasps. **Step 16000** pushed the knight to about e5 and tipped it over.

### 17:37: LoRA step 500 runs away, and the stall guard couldn't trip

- **Runaway:** commands went to shoulder_lift 844° and elbow 374°. The model chains each plan onto **its own
  previous commands**. Once the clamped arm stopped following, that history drifted further from reality with
  every plan and grew without bound. The elbow then pressed about 5° from its goal for about 20 s, and the
  servos faulted again.
- **Stall-guard bug:** the guard tripped at a 6° gap, but the 5°/tick cap keeps the goal within about 5° of the
  arm. With both defaults on, **the guard could never trip.** It only caught the table hit because that run used
  a 10° cap.
- **Fixes** (`543ce89`):
  1. Stall threshold 3° (the CLI rejects a threshold ≥ the cap).
  2. **Runaway abort** when a command goes more than 30° outside the joint limits.
  3. The history records the goal **actually sent**, not the raw command.

  Replaying all real-arm logs through the new defaults: the table-hit run stops at 4.9 s, the runaway at 5.1 s,
  and the good ACT run never trips.
- **Guard false positives fixed:**
  - At rest, ACT asks for a shoulder position the folded arm can't reach. That doesn't count when the goal is
    beyond the demos' reached range.
  - At 19:19, base FLUX tucked the folded elbow further in and tripped after 0.8 s. Folding further at rest isn't
    a stall (`f2a9dc7`).

### 17:31–19:11: why zero-shot FLUX fails, and the normalization fix

Full analysis in [flux_joint_convention.md](flux_joint_convention.md). In short:

- For the 5 arm joints, FLUX predicts per-tick **deltas** added to the state we send. It never outputs absolute
  angles in "its" convention. What matters is where our values land after its q01/q99 normalization.
- The shipped normalization equals one training dataset's ranges exactly (`VoicAndrei/so100_kitchen`, old SO-100
  calibration: shoulder_lift 85…184°). Ours is −97…39°, so our rest pose normalizes to about −2.8 to −4.7 on a
  −1…1 scale. Per-tick ranges were also 4–5× ours on pan, wrist_flex and wrist_roll.
- BFL normalized **each training dataset by its own quantiles**. The ~1,400 community SO-100/101 datasets share
  our calibration convention (only wrist_roll's zero differs). shoulder_lift's sign is flipped between MolmoAct
  data and community data.
- **Fix:** serve with our own dataset's quantiles (`--normalization`, `ff53676`; episodes 0/1 excluded for
  wrist_roll). Normalization happens **only on the server**. The client always sends raw degrees and gripper %.
- **GPU check** on held-out episodes 24–30:

  | Policy | Arm / gripper error | Right direction on moving windows |
  |---|---|---|
  | Base, checkpoint normalization | 13.36° / 6.40 % | 53 % |
  | Base, our normalization | 6.76° / 2.84 % | 45 % |
  | Hold the last command | 4.00° / 1.90 % | 0 % |

  So the whole-episode metric mainly rewards not moving. Judge checkpoints on **moving windows**
  (`scripts/flux_convention/motion_windows.py`).
- **Real arm, base FLUX with our normalization (19:15, 19:21):** safe and bounded (shoulder_lift −118…−38°
  instead of 844°). It turned toward c3 and opened the gripper, and made one grasp attempt at about 40 s of wall
  time. But it kept the shoulder folded (−97°, where the demos go to about +6°), and pan was pinned at the −18°
  limit.

### 19:25–20:44: ownstats LoRA and the ACT comparisons

- `Gawtam/flux3-chess-lora-ownstats` (step 125, trained with our quantiles, read from the adapter's processor
  files): 6.7–6.9° offline (card 6.3°). Two runs: partial reaches, alternating with returns to rest, and no grasp.
  The second run's wrist_roll drifted about 85°.
- **ACT, 15 episodes / 4000 steps** (6 runs) and **10 episodes / 16,000 steps** (3 runs): all safe at about
  0.44 s per plan, 26.5 Hz. More episodes helped more than more steps. The 10-episode model learned reach and
  grasp but not the release at d5. **Offline error didn't separate the models; the real arm did.**
  → [act_inference_experiments.md](act_inference_experiments.md).
- A held grasp shows up in the logs as the gripper stopping at about 5.5–6.5 % while commanded to about 1–3 %. An
  empty close reaches the commanded value. That is the only grasp signal in the joint logs; whether the piece
  reached d5 needs eyes or the cameras.
- "Clamped ticks" in the run summary only counts joint-limit clips. Counting the 5°/tick cap as well inflated
  one report (474 vs 49).
- One transient `no status packet` on servo 3 at connect; the retry seconds later worked.

### 21:14–21:40: ownstats-4gpu step 500 and flux3-chess-lora step 2500

- **ownstats-4gpu step 500** (`6cc8a2d`): 5.3–5.6° offline, with aimed reaches to c3 and the gripper open. All
  three runs were stopped by the stall guard: run 1 pressed into the board with the wrist about 20° low, and runs
  2/3 alternated between reach and rest, with one close on nothing at pan 0°.
- **flux3-chess-lora step 2500** (`--no-normalization`): 3.0–3.3° offline. **The first FLUX run to reach grasp
  height** (shoulder_lift +14°, elbow 8–15°), with the gripper opening at the piece. Both runs were stopped by the
  stall guard at the piece before any close.
- **A relaxed-guard run was blocked** by Claude Code's auto-mode permission check (stall guard off, limits +5°).
  The user ran it with `!` instead. Both of those runs (`20261017_213819`, `_214005`) crashed mid-episode, the
  second with the arm stretched out, and left the follower bus faulting.

### 22:16: recording crash, an empty episode

- `You must add one or several frames with add_frame before calling add_episode`: → was pressed twice, ending
  the episode and then the reset. The second press was still latched when episode 2 began, so it ended with 0
  frames. Episodes 0–1 (699 frames) were intact on disk. Press → once, then wait for "Recording episode N",
  especially with a 5 s reset. Resume with the full repo id and `--episodes <remaining>`.

### 22:29–23:03: ownstats-4gpu step 2500, bus faults and the split pick-and-place

- **Offline:** 2.8° / 3.1° (card 2.59°), half of step 500.
- **Arm bus:** connect failed on motor 5, then motor 1, while a read-only check in between was fine. That came
  after the relaxed-guard crashes. **Reconnecting/power-cycling the follower fixed it.**
- **The Space was `SLEEPING`, not `PAUSED`,** so `deploy` waited forever for it to wake and it had to be restarted
  by hand.
- **Three back-to-back 20 s runs (22:53–22:55)** covered reach, grasp (gripper held at 6.4 % vs 2.7 %), carry and
  release, *split across three episodes* by the stall guard and the 20 s limit. One gripper overload error came
  from squeezing.
- **40 s run with `--stall-gap 4.5`:** two pick attempts, one carry to d5, but the grip slipped before the carry.
  wrist_roll drifted about 27°.

### 23:03–00:23: median community normalization (multimodalart sim Space)

- Stats from `multimodalart/flux-3-action-so101-sim@09edebb` ("median new-calibration SO-101 dataset stats"). The
  convention matches ours, except **wrist_roll**: our arm rests at about −78°, the community median at about 0°,
  so 85 % of our frames fall outside the model's normal range. A prediction also moves pan and wrist about 2×
  more per tick than our demos.
- **Base FLUX + median stats:** 7.3° offline (ours: 6.8°), 47 % right direction (chance). Two real-arm runs:
  safe, an elbow-only reach, one empty close, and a stall at 13.6 s. Not better than our own stats.
- **flux3-chess-lora step 2500 + median stats:** the best offline FLUX number at that point (71 % right direction
  on moving windows, beating hold 6.1° vs 7.1°). On the arm the **runaway abort fired at 7.3 s** (shoulder_lift
  commanded −134°, below the folded floor). The adapter was trained on the checkpoint normalization, so mixing in
  different stats is out of distribution.
- **Every dry run ends on the runaway abort** (the arm doesn't move, so the model's self-chained commands drift).
  That's expected, not a fault.

### 00:59–04:19: more datasets pushed to the Hub

- Pushed with `scripts/push_dataset.py` in a loop, since the script takes one dataset per run and the glob only
  matches inside `~/.cache/huggingface/lerobot/`:
  - `chess_dataset_c1*` (7 datasets), `d1*` (7), `new_b1*` (7), `new_c1*` (7) and `new_d1*` (7)
  - each set is one source square (rank 1) to the other squares in its file
- Two empty folders (`c1→c7 …_005143`, `new_c1→c6 …_032121`) have only `meta/info.json`. They are aborted
  recordings that were redone seconds later, and pushing them fails with `Repository Not Found`. Safe to delete
  locally.
- Episode counts differ: the `new_b1` set has 6 or 8 episodes per dataset, the others 7 or 8.

### 02:09–02:26: ownstats-4gpu-ext step 4500, the full pick-and-place in one episode

- `Gawtam/flux3-chess-lora-ownstats-4gpu-ext` (`a555c08`, the same quantiles as ownstats-4gpu). Offline: 2.83° arm,
  and **3.86° vs 7.12° for hold on moving windows, 87 % right direction**.
- **02:14:** missed the first grasp, then re-gripped, carried and released at d5. **02:17: a clean grasp on the
  first try, carry, release at d5.** Both runs ended on the stall guard *after* the release: the stretched-out
  arm can't lift back. **The first complete pick-and-place in a single episode with a FLUX policy** (judged from
  the gripper readings; nobody checked the board).
- A no-guard run (all four guards off) was blocked by auto-mode. The user ran four `_noguard` attempts
  (02:22–02:26), and none ended with an episode result.
- This write-up is on branch `worktree-rollout-ownstats-4gpu-ext` only (see
  [below](#branches-not-on-rollout)).

### 03:00–06:43: scene-camera drift watch

- A `/loop` job registered each new `chess_dataset_new_*` dataset's first scene frame against
  `new_b1→b2` (feature matching, with the arm strip masked) and also compared the first and last frame within each
  dataset. A shift over 3 px warns.
- All 21 `new_*` datasets stayed **under 0.8 px**, with rotation under 0.05°. The same check on the older
  `chess_dataset_{a,b,c,d}*` sessions finds **90–245 px shifts and 4–7° rotation**, so **the old and new series
  have different scene-camera poses**. Don't mix them in training without accounting for that.
- A file that is still being written can show a transient within-dataset shift (0.89 px → 0.21 px once
  finished).

### 01:50 and 04:21: FLUX's imagined future, and `--debug`

- FLUX denoises about 1.4 s of future video together with each 42-action chunk (11 latent frames → 41 frames,
  256×512 scene|wrist canvas), and `_sample` throws it away. The VAE decoder ships with the policy.
- **`run --debug`** (`5ca9787`): records every frame, each plan's inputs and full output, and the decoded imagined
  future. `python -m rollout view [run] --open` builds a self-contained `viewer.html` (findings, imagined vs
  actual, joint charts, plan table). It adds about 0.2 s of GPU per plan (requests about 2.1 s instead of 1.5 s).
  A test runs BFL's real sampler and confirms that debug mode returns identical actions.
- Viewer fixes: decode in the VAE's dtype, never fail a plan on a decode error (`fab9d27`), and image sizing.
  Snap Firefox can't open files in hidden directories, so copy a run elsewhere to preview it.

### 04:43–06:15: 28-move LoRA debug runs

`Gawtam/flux3-chess-lora-28moves`, with its own normalization. Details are in
[inference.md](inference.md) ("28-move LoRA debug runs").

- **Joint limits must come from the move's own dataset.** b1 needs pan −24° and a1 needs −27°, while the default
  c3→d5 limits stop at −18°. Use `--limits-from <that move's dataset>`.
- **Step 1000, b1→b4 / a1→a4:** reaches to the piece were well aimed, and the imagined frames matched the real
  ones. The stall guard stopped every run at 3.2–3.9 s with the fingers at the piece. In the a1 run the next plan
  would have **backed off with the gripper still open**: the model never commanded a close at the piece.
- **Prompt format:** "Move the object from b1 to b4" vs the trained "Move the figure from b1 at 2 cm to b4 at 8 cm".
  The chunks differed by up to 19°, but the overall error was similar.
- **Step 2000 lives in a subfolder** (`checkpoints/002000/pretrained_model_ema`). The loader now accepts an
  adapter folder inside its repo (`b026e79`). The run went the full 20 s with three reaches and the first gripper
  close, on nothing. wrist_roll sat on its −112° limit from 4 s, which may have kept the fingers misaligned.
- **How closely does the arm follow the plan?** Within 2–4° (shoulder_lift and elbow, mostly from the 5°/tick
  cap). Consecutive plans join within about 0.4°. **The real problem is stop-and-go:** the synchronous loop
  holds the arm still about 2.1 s per plan, so the arm waited **67 %** of the wall time, and each new history
  looks like an arm at rest.
- **`--n-action-steps` for FLUX** (`b1a38eb`; `space deploy --n-action-steps N`, 1–42): the server executes the
  first N of the 42 predicted actions. With N = 10 the first grasp attempt came at 5.5 s
  instead of 17 s, but closed just off the piece, and waiting rose to **86 %** of wall time.
- **Step 3000, 10 actions/plan, a1→a4 (06:10):** the full 20 s, 60 plans, 12 clipped ticks, a complete
  pick-and-place *motion* twice. But the jaws were empty both times, and the knight stayed upright on a1. The
  likely cause, inferred from the frames and not proven: the fingers stop at shoulder_lift about −8°, where the
  demos go to +3…+12°. The user's own reruns: 06:15 was stopped by the stall guard after 26 plans; 06:17
  (prompt "d1 → d4") recorded no episode.
- `docs/inference.md` now has a **run log index** of all 61 runs, and every run has a `viewer.html`.

### 06:22–06:36: the Space kept an old setting

- After `deploy --default-action-steps` deleted `ROLLOUT_N_ACTION_STEPS`, the running Space **kept serving
  `[10 steps/plan]`** across a restart for about 11 minutes (about $0.50 of idle billing). Fix (`15cb347`): set
  the variable to `0` ("use the model's default") instead of deleting it, and apply deletions before additions.
  This fix hasn't been exercised against the live Space yet.
- The Space was left paused, configured for ownstats-4gpu-ext, ready for a c3→d5 run with
  `--task "Move the figure from c3 at 6cm to D5 at 10 cm"`. (A session started at 06:35 to run it; no log from it
  was found in `rollout_logs/`.)

## Debugging lessons

**Data collection**

- A LeRobot dataset can't prove its own frame rate (`timestamp = frame_index / fps`). Use `record_timed.py`'s
  report, or frame count against wall time.
- The Rerun viewer is the main cause of dropped frames. Keep it off, and use V4L2 + MJPG cameras.
- Resuming overwrites that dataset's timing JSON (it did for episodes 0–20 of the main dataset).
- Press → once per phase. Ctrl+C during a reset loses the just-recorded episode; → or Esc saves it.

**Evaluating policies**

- Evaluate offline **teacher-forced**. Replaying recordings with the model's own commands as history makes the
  errors compound and look like a broken model.
- Whole-episode error rewards standing still ("hold" scored 4.0°). Rank checkpoints on **moving windows**
  (right direction, motion covered), and ultimately on the arm: offline error didn't separate ACT models that
  behaved very differently on the arm.
- The gripper's measured vs commanded gap (about 5–6 % vs 1–3 %) is the logs' only grasp signal. Placement needs
  the cameras, so use `--debug`.

**Safety layer** (defaults: demo-range limits, 5°/tick cap, stall guard 3°, runaway abort at 30° beyond the limits)

- Thresholds interact. A stall gap at or above the per-tick cap can never trip, and the CLI now rejects it.
- Clamped commands must be what goes into the history, or a clamp snowballs into a runaway.
- The stall guard needed exceptions for the folded rest pose. It still stops runs at legitimate contact (fingers
  on the piece or board, a stretched-out arm that can't lift). `--stall-gap 4.5` (it must stay below 5) helps.
- Dry runs always end on the runaway abort. That's expected.
- **The arm goes limp when a run ends** (torque off). A run that ends mid-air leaves the next one starting from a
  different pose, so reset to rest between runs.
- Claude Code's auto-mode blocks guard-off runs on hardware. The user can run them with `! <command>`.

**Hardware**

- `no status packet` at connect is usually transient; retry. After a stall or overload (servos stop
  responding, even torque-off fails), **power-cycle the follower**.
- A knocked-over piece invalidates the run, so check the wrist camera before each run.

**Space and serving**

- Normalization is server-side only (`ROLLOUT_NORMALIZATION` / `--normalization`). A LoRA must be served with the
  normalization it was trained with. Mixing in other stats ran away.
- `SLEEPING` ≠ `PAUSED`: a sleeping Space may not wake on deploy, so `start` it. A monitor can read a stale stage
  right after a restart. A deleted Space variable can linger in a running Space.
- Python buffers stdout when writing to a log file, so flush progress lines.

**Claude Code in this repo**

- New worktrees branch from `origin/main`, which doesn't contain `rollout/`. Reset the worktree branch to
  `rollout` first (this happened in five sessions).
- The worktree guard rejects heredocs, compound `cd … &&` commands and computed paths. Put the logic in a script
  file instead.
- Sessions have no terminal, so the keyboard stop (Esc) isn't available. The power switch or Ctrl+C is the stop.
- Job temp dirs (`~/.claude/jobs/<id>/tmp`) are deleted with the job. Analysis scripts worth keeping must go into
  `scripts/` (the drift check `drift.py` and `check_fps.py` were left there).

## Scoreboard

Offline: held-out episodes 25/30 of the c3→d5 dataset, mean arm error (lower is better; hold the last command ≈
4.0°, and about 7.1° on moving windows).

| Policy | Offline arm error | Best real-arm result |
|---|---|---|
| ACT `act_chess_c3_to_d5` (31 ep) | 1–7° (trained on those episodes) | full pick-and-place, back to rest |
| ACT step 2000 (31 ep) | — | knight on d5, home in about 11 s (1 run) |
| ACT 15 ep / 4000 | lift 23°, elbow 19° (ep 25) | sequence 6/6, weak release 5/6 |
| ACT 10 ep / 16,000 | lift 22°, elbow 18° (ep 25) | reach and grasp, no release at d5 |
| FLUX base, checkpoint norm | 13.36° | — |
| FLUX base, our norm | 6.76° | turn toward c3, open, one empty close |
| FLUX base, median norm | 7.3° | elbow-only reach, empty close |
| flux3-chess-lora step 250 / 500 | card 8.80° / 6.72° | table hit / runaway (844°) |
| flux3-chess-lora step 2500 | 3.0–3.3° | grasp height, gripper open (stall stop) |
| …step 2500 + median norm | 5.0° (71 % direction) | runaway at 7.3 s |
| ownstats step 125 | 6.7–6.9° | partial reaches |
| ownstats-4gpu step 500 | 5.3–5.6° | aimed reach, open gripper |
| ownstats-4gpu step 2500 | 2.8–3.1° | reach, grasp, carry, release split over 3 episodes |
| **ownstats-4gpu-ext step 4500** | **2.83°; 3.86° on moving windows, 87 %** | **pick-and-place in one episode, twice** |
| 28moves step 1000 / 2000 / 3000 | — (other moves) | at the piece; full motion with empty jaws (step 3000) |

## Open problems

1. **Stop-and-go execution.** The arm waits 67 % of wall time at 32 actions/plan and 86 % at 10. Overlapping the
   next request with execution (asynchronous replanning, dropping stale actions) is the suggested fix and isn't
   built yet.
2. **Grasp height and alignment for FLUX.** Closes land just off or above the piece (lift about −8° vs the demos'
   +3…+12°), and wrist_roll drifts or pins at its limit.
3. **The stall guard ends runs at legitimate contact** and after a release. It needs a contact-aware rule
   rather than a looser gap.
4. **Placement isn't verified.** Most "success" calls come from gripper readings. Use `--debug` frames or a scene
   check before and after each run.
5. **The scene camera moved** between the old and new recording series.

## Branches not on `rollout`

Checked against `origin/rollout` at `15cb347`:

| Branch | Commits | What | Note |
|---|---|---|---|
| `worktree-rollout-ownstats-4gpu-ext` | `bbf4cfb` | write-up of the **best FLUX result** (step 4500, one-episode pick-and-place) and `ownstats4gpu_ext_norm.json` | should be merged |
| `worktree-rollout-ownstats-4gpu-2500` | `7a49c78`, `8b71685`, `3240adc` | ownstats-4gpu step 2500: offline check, bus errors, three runs, 40 s run | should be merged |
| `worktree-act-rollout` | `fb09748` | `--policy local:act` and the ACT step 2000/4000/16000 runs | superseded by `local:lerobot`; its run results aren't in the docs |
| `worktree-flux-norm-stats` | `4e4d79f` | `--norm-stats` / `scripts/flux_norm_stats.py` | superseded by `--normalization` |
| `main` | `081cfd0` | merge of PR #1 | `rollout` hasn't been merged back to `main` since |

## Space spend

The local ledger (it deliberately over-counts, including sleeping time) went from $0.45 after the first deploy to
about **$17.24 of $35** at 06:36 on 2026-10-18. The budget was raised from $20 to $35 at about 06:07. Real charges:
https://huggingface.co/settings/billing.
