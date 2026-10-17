# FLUX 3 Action and our joint calibration (analysis, 2026-10-17)

**Question.** The zero-shot FLUX 3 Action SO-101 model produced nonsense for our arm, and we suspected
that our calibration (lerobot 0.6.0) differs from the one the model was trained on. Is there a mapping from
our joint values to the ones the model expects, and what is it?

**Short answer.** The difference isn't a fixed offset between calibrations. It's **normalization**. BFL
normalized every training dataset with **its own** joint range, so the model learned joint values rescaled to
about −1…1 per dataset. The released checkpoint then ships **one fixed** range (from MolmoAct-style
datasets), which puts our values far outside anything the model saw. The mapping that reproduces training
is: **normalize with our own dataset's quantiles** (`--normalization`).

**Verified on the GPU (section 7):** with our normalization the zero-shot base model's error on held-out
episodes halves (13.4° → 6.8° arm, 6.4 → 2.8 % gripper). But on the windows where the demonstrator actually
moves, **no FLUX variant predicts the motion better than chance**: zero-shot, or the LoRA at steps 250/500. **"Hold the last
command" beats all of them** on the error metric the LoRA card uses. The mapping fixes the input; it doesn't give the
model the task. ACT, trained on our demonstrations, tracks the motion (100 % right direction), though on episodes it was trained on.

## 1. What the model can and can't tell us

The first idea was to run the model on our images and task text and read off the joint angles it
predicts, in its own convention. That doesn't work directly:

- For the 5 arm joints the model predicts **per-tick changes** (deltas). The reference code adds them onto
  the **command history we send in**, so the absolute angles it returns are always in our convention.
  Only the gripper is predicted as an absolute value.
- What the model *does* depend on is how our values look **after normalization**: the state and past
  commands are normalized with the checkpoint's quantiles before they reach the network. So the right
  question is "what normalized values did the model see for a given arm pose?". The training data answers that.

## 2. Our calibration: 31 episodes, 13,229 frames

`niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741`, all frames (degrees, gripper %):

| Joint | q01 … q99 | Rest pose (frame 0 of each episode, mean ± sd) |
|---|---|---|
| shoulder_pan | −14.6 … 3.8 | −0.9 ± 3.6 |
| shoulder_lift | −97.1 … 39.2 | −92.6 ± 11.6 |
| elbow_flex | −24.3 … 96.5 | 92.8 ± 5.2 |
| wrist_flex | 55.4 … 99.8 | 74.0 ± 4.9 |
| wrist_roll | −104.8 … 45.8 | −77.6 ± 16.0 |
| gripper | 0.8 … 35.9 | 3.5 ± 4.5 |

The arm starts every episode folded at rest: shoulder_lift at the bottom of its range, elbow at the top.

## 3. The model's checkpoint normalization doesn't match any raw dataset

The released checkpoint normalizes the state with fixed quantiles (`policy_preprocessor_step_3_*.safetensors`,
`dataset_statistics.json`): shoulder_lift **84.8 … 184.2**, elbow **61.4 … 178.9**, pan −92 … 54,
wrist_flex −24 … 81, wrist_roll −125 … 60, gripper 0.5 … 54.

Then I checked the SO-101 data the model card names, `lerobot/community_dataset_v3` (1,755 sub-datasets),
grouped by `robot_type`. These are the medians across sub-datasets of their own `stats.json`, plus the rest pose from
frame 0 of about 600–900 episodes per group:

| Group | Datasets | shoulder_lift q01 … q99 | Rest pose (pan, lift, elbow, wflex, wroll, grip) |
|---|---|---|---|
| `so100` (`main_*` names) | 895 | −99.3 … 50.1 | 1.9, −94.5, 82.6, 50.9, −0.1, 0.4 |
| `so101` (`main_*` names) | 312 | −102.2 … 46.7 | 0.6, −99.9, 85.5, 65.7, −4.1, 0.6 |
| `so101_follower` | 121 | −99.0 … 52.6 | −2.8, −98.8, 98.0, 67.0, −0.9, 2.1 |
| `so100_follower` | 117 | −100.0 … 42.1 | −0.2, −98.1, 99.0, 47.3, 1.8, 1.0 |
| **ours** | 1 | −97.1 … 39.2 | −0.9, −92.6, 92.8, 74.0, −77.6, 3.5 |

**All of them use the same convention as ours** (rest pose: lift ≈ −95…−100, elbow ≈ 83…99). The only
notable difference is **wrist_roll**: their rest is ≈ 0°, ours ≈ −78°. None of these groups matches the
checkpoint's lift 85…184. So the gap isn't in community_dataset_v3.

## 4. How BFL actually built the training data

BFL publishes the training corpus definition: [`configs/so101/community_corpus.json`](https://github.com/black-forest-labs/flux-action/blob/830f1ce178d7aea52f18fde31d8e8e36ce6bae21/configs/so101/community_corpus.json)
in `flux-action`. Three facts in it explain everything:

1. **Sources**: 1,016 training parts. **709 from `allenai/MolmoAct2-SO100_101-Dataset`** (read from each
   original community repo) and **307 from `lerobot/community_dataset_v3`**, 631,391 training windows.
2. **Per-dataset normalization**: *"x_norm = clip(2·(x − q01)/(q99 − q01) − 1, −6, 6) with the part's own
   q01/q99"*, for the state and for the action targets alike. The model never saw raw degrees, only each
   dataset's joints rescaled to about −1…1 over **that dataset's own range**.
3. **Action targets**: per-tick deltas `a[t] − a[t−1]` on the 5 joints (first row 0), gripper absolute.

The MolmoAct parts store their joints differently: shoulder_lift ranges about 60…195 (e.g.
`356c/so100_duck_reposition_1`: lift 66.8…195.3, elbow 73.7…180.7). That's where the checkpoint's fixed
quantiles come from.

## 5. The one thing normalization can't absorb: sign

Per-dataset normalization removes offsets and scales, but not a **flipped direction**. So for 54 sampled training
parts I read frame 0 of their episodes from the source repos and placed it in that part's normalized range
(−1 = its q01 end, +1 = its q99 end):

| Rest pose, normalized | pan | **lift** | elbow | wflex | wroll | grip |
|---|---|---|---|---|---|---|
| MolmoAct parts (30), median | −0.13 | **+0.99** | +0.99 | +0.42 | +0.15 | −0.99 |
| community_dataset_v3 parts (24), median | −0.02 | **−0.99** | +1.00 | +0.55 | −0.42 | −0.99 |
| **Ours** (our own q01/q99) | +0.48 | **−0.93** | +0.94 | −0.16 | −0.64 | −0.85 |

- **shoulder_lift has opposite signs in the two halves of the training data.** In MolmoAct parts the folded
  arm is at the top of the range (+1, raw ≈ 189), in community_dataset_v3 parts at the bottom (−1, raw ≈ −98).
  The model was trained on both.
- elbow (+1) and gripper (−1) agree everywhere. Pan and the wrist joints have no consistent rest position, so there's
  no sign to detect.
- **Normalized with our own quantiles, our data looks exactly like a community_dataset_v3 training part**
  (lift −0.93, elbow +0.94, gripper −0.85). That's about 30 % of the training parts, so it's in distribution.

What the released checkpoint does instead: our rest pose (lift −97) normalized with lift 84.8…184.2 gives
**2·(−97 − 84.8)/99.4 − 1 ≈ −4.7**, and elbow 93 gives ≈ −0.47 instead of +1. The model sees a folded arm as
an impossible pose, which explains the erratic zero-shot behaviour.

## 6. The mapping: our own quantiles, computed like BFL's

`rollout/normalization.py` computes BFL's statistics for any LeRobot dataset: percentiles 1 and 99 over
all frames, of the measured state and of the action targets (per-episode deltas, gripper absolute). For our
dataset:

| Joint | Our state q01 … q99 | Checkpoint state | Our action (delta) q01 … q99 | Checkpoint action |
|---|---|---|---|---|
| shoulder_pan | −14.5 … 3.8 | −92.0 … 54.1 | −0.79 … 0.62 | −3.60 … 3.34 |
| shoulder_lift | −97.1 … 39.2 | 84.8 … 184.2 | −3.52 … 3.78 | −2.64 … 4.13 |
| elbow_flex | −24.3 … 96.5 | 61.4 … 178.9 | −2.55 … 3.69 | −3.43 … 4.13 |
| wrist_flex | 55.4 … 99.8 | −24.0 … 80.6 | −1.49 … 1.32 | −5.10 … 5.45 |
| wrist_roll | −104.8 … 45.8 | −124.5 … 60.2 | −0.97 … 0.79 | −3.25 … 3.16 |
| gripper | 0.8 … 35.9 | 0.5 … 53.6 | 0.65 … 35.95 (absolute) | −0.10 … 54.16 |

`FluxModel(normalization=...)` replaces the checkpoint's quantiles with these. flux_action reads them on every
call (`processing/normalization.py`, nothing cached), so the state, the past-command conditioning and the
de-normalization of the predicted deltas all use our ranges. A side effect that helps safety: a "full speed"
prediction for pan becomes ±0.7°/tick (our demos) instead of ±3.5°/tick.

Why server-side and not a client-side remap: state and deltas are normalized with **different** ranges,
so no single per-joint affine transform applied on the client can reproduce both.

```bash
.venv/bin/python -m rollout.normalization niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741 -o chess_norm.json
.venv/bin/python -m rollout.server.space deploy --no-adapter \
  --normalization niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741 --revision c9e13b2aca6a0a472b3ea03fd90cff32d3e85849
```

`rollout/tests/test_flux_equivalence.py::test_dataset_normalization_replaces_checkpoint_quantiles` checks
on BFL's model code that passing the checkpoint's own quantiles changes nothing and ours changes the plan.

**This applies to the zero-shot base model.** The `Gawtam/flux3-chess-lora` adapter was fine-tuned with the
checkpoint's fixed quantiles, so it learned our values in that normalization. It should keep it, not ours.

## 7. Verification on the GPU

Setup: the Space (`a100-large`), base model `black-forest-labs/flux-3-action-so101` @ `c9e13b2`, no adapter, run
twice: once with our normalization, once with the checkpoint's. Teacher-forced dataset replay
(`python -m rollout run --robot dataset`) on the held-out episodes 24–30 (3,049 frames, 100 plans).
Same evaluator, same episodes for every row.

### 7.1 Whole-episode error (the metric the LoRA card reports)

Mean |predicted − recorded command| over every executed tick, frame-weighted (`summarize_eval.py`):

| Policy | pan | lift | elbow | wflex | wroll | **arm** | gripper |
|---|---|---|---|---|---|---|---|
| FLUX base, checkpoint normalization | 10.91 | 18.32 | 15.84 | 13.98 | 7.76 | **13.36** | 6.40 % |
| **FLUX base, our normalization** | 3.03 | 12.39 | 11.90 | 4.85 | 1.61 | **6.76** | 2.84 % |
| *LoRA card: base / step 500 (88 windows, 24–30)* | | | | | | *12.49 / 6.72* | *5.82 / 4.24 %* |
| **Hold the last command** (`local:hold`) | 1.21 | 8.14 | 6.23 | 3.68 | 0.76 | **4.00** | 1.90 % |

- Our evaluator reproduces the card's base number (13.36° vs. 12.49°), so the comparison holds up.
- **Our normalization halves the base model's error** on every joint. Zero-shot it matches the card's fine-tuned
  step 500 (6.76° vs. 6.72°).
- **But "hold the last command" beats everything.** Under teacher forcing every plan starts from the
  demonstrator's last command. Most of each episode is slow or still, so not moving scores 4.0°. So this
  metric can't show whether a model learned the task. Its low numbers mostly mean "didn't move much".

### 7.2 Windows where the demonstrator moved (`motion_windows.py`)

Only the 32-tick plans in which a recorded arm command moves at least 10°. Two questions per joint that moved
at least 5°: did the model move it **the right way** (at least 10 % of the recorded motion, same sign), and how much of the
motion did it **cover** (1 = all of it, 0 = none, < 0 = the wrong way)?

| Policy | Episodes | Moving windows | Arm error | Hold error | Right direction | Median coverage |
|---|---|---|---|---|---|---|
| FLUX base, checkpoint normalization | 24–30 | 48 | 16.22 | 6.93 | 53 % | 0.74 |
| FLUX base, our normalization | 24–30 | 48 | 8.79 | 6.93 | 45 % | −0.19 |
| FLUX + LoRA step 500 (eval stopped partway) | 24, part of 25 | 23 | 10.11 | 7.58 | 54 % | 0.56 |
| FLUX + LoRA step 250 | 0, 25, 30 | 23 | 12.45 | 7.40 | 49 % | −0.04 |
| ACT `rtrtwrw/act_chess_c3_to_d5` (*trained on these episodes*) | 0, 25, 30 | 7 | **2.09** | 6.78 | **100 %** | **0.97** |
| Hold | 24–30 | 48 | 6.93 | 6.93 | 0 % | 0.00 |

- **No FLUX variant predicts the motion**: about 50 % right direction is chance, and all are worse than not moving.
  With our normalization the base model mostly stays put (coverage −0.19). With the checkpoint's it moves a lot
  (0.74) but just as often the wrong way, hence the larger error.
- The LoRA at steps 250 and 500 hasn't learned the motion yet either, consistent with how the step-500 arm run went.
- ACT follows the demonstrations closely (100 % direction, 97 % coverage), but these are training episodes for it,
  so this is an upper bound. On the real arm it ran a complete, demonstration-like episode (`docs/inference.md`).

### 7.3 Conclusions

1. **The calibration mapping is solved**: normalize with our own quantiles, exactly as BFL's pretraining
   treated every dataset. It removes the out-of-range inputs (rest pose at −4.7 → −0.93) and halves the error.
2. **It doesn't make FLUX zero-shot useful for this task.** That's a capability gap, not a calibration bug.
3. **Fine-tuning is the way to get a FLUX policy.** Two concrete suggestions for the next LoRA run:
   - train with **our** normalization (`rollout/normalization.py` gives the quantiles), so the inputs look like
     pretraining instead of out-of-range values the adapter must first learn around;
   - select checkpoints by the **moving-window** metric (beats hold? right direction?), not by whole-episode
     error, which "hold" wins.
4. **For the arm today, ACT is the policy to use**, fully local (`--policy local:lerobot`).

Cost of this verification: about $0.60 of A100 time. Total on the Space so far: about $2.09 of the $20 budget (the
ledger's estimate errs high).

## 8. On the real arm: base FLUX with our normalization (19:15)

Space with the base model and our normalization; one 20 s episode with the safety defaults (demo-range limits,
5°/tick, stall guard, runaway abort at 30°). Logs: dry run `20261017_191515`, motion run `20261017_191557`.

- **Dry run** (no goals sent): plan 1 settled gently at rest (lift −97 → −103, pan −6 → −9.5). The following
  plans looked like the start of a demonstration: lift rising, elbow extending, pan to about −18°, gripper
  opening to 27 % at tick 96. The runaway abort stopped it at tick 186 (elbow commanded −65°), the expected dry-run
  drift since the arm never followed.
- **Motion run: completed, safely.** 600 ticks (20 s of motion over 49 s: 1.07 s moving, about 1.5 s waiting per plan). No
  stall, no runaway, no servo fault. The arm followed every goal within the 5° cap. Commands stayed bounded
  (shoulder_lift −118…−38°, elbow 19…117°), unlike the LoRA runs (844°).
- **What it did:** rest for about 4 s. Then it swung pan to −17.5° (toward c3) and opened the gripper to about 28 %, with lift
  up to −66°: an approach. From about 10 s on it **never grasped and never lifted**. shoulder_lift kept falling back to rest
  height (−96°) and rising again, the elbow swept 20–97° (arm low, reaching forward), the gripper stayed
  15–30 % open, and wrist_roll drifted 70° (−89 → −17°, far from the demos' ≈ −78°). Clamps: mostly pan wanting to go past
  −18° (263 ticks; demos stay within −17…8°) and elbow/lift pressing at the rest position.
- **Reading:** the normalization keeps zero-shot FLUX in a sensible, safe range, and it starts the right
  kind of motion (toward the piece, gripper open). But it doesn't complete a grasp, as section 7.2 predicted. The
  stop-and-go execution (1.5 s pauses) likely makes it worse: after each pause the history no longer looks like smooth 30 Hz motion.

### 8.2 Second attempt, piece on c3, arm at rest (19:19 and 19:21)

- **19:19, false stop after 0.8 s.** The arm started folded with the elbow at 85° (run 1: 96°; demos' rest
  92.8 ± 5.2°). Plan 1 gently tucks the elbow further in (goal 92–94°); the folded elbow can't follow (85 → 87°) and
  the stall guard read the 5° gap as blocked. That's the arm pressing into its own fold, not a collision. The
  reach-edge exemption didn't apply because 94° is inside the demos' reach. **Fix:** with the arm folded (shoulder_lift
  within 10° of the lowest demonstrated value), pushing further into the fold (lift down, elbow up) isn't a stall.
  Replayed on all five real-arm logs: the two LoRA runs still stop at about 5 s, the ACT run and both base-FLUX
  runs never trip, and this false stop is gone (`test_stall_guard_ignores_folding_further_at_rest`).
- **19:21, full run, completed safely.** 600 ticks (20 s of motion over 48.7 s), 19 plans at 1.50 s each, no stall, no
  runaway, the arm within 5.5° of every goal. Commands bounded (lift −113…−70°, elbow 20…98°). Logs: `20261017_192112`.

  | Time | Joints (state) | Reading |
  |---|---|---|
  | 0–7 s | pan −6 → −15°, lift −97 → −80°, elbow 86 → 68° | heads toward c3 |
  | 9–40 s | gripper opens 25 → 37 %; pan pinned at the −17.5° limit (asks for down to −23°); lift −97…−74°; elbow 20–55°; wrist_flex 80–95° | reaches low, gripper open and pointing down |
  | **≈ 40–43 s** | **gripper closes 36 → 2.5 %** | **a grasp attempt** |
  | 43–49 s | gripper reopens to about 12 %; lift −97 → −82° | |

  Compared with the demonstrations: at the grasp the demos **raise the shoulder** (lift ≈ +6°, elbow ≈ 26°, pan
  ≈ −14°, e.g. the ACT run). FLUX keeps the shoulder folded (≈ −97°) and reaches with the elbow, so the gripper ends up
  much closer to the base, likely short of c3. wrist_roll drifted again (−86 → −8°; demos ≈ −78°). Pan spent 281
  ticks pinned at our −17.5° limit while the model asked for more.

**Overall (two full runs):** zero-shot FLUX with our normalization is safe on our arm, and it does the coarse
structure of a pick (turn toward the piece, open, close once). But with the wrong arm configuration (shoulder
folded instead of raised), and with no carry or place. A fine-tune with our normalization (section 7.3) is the
next step for FLUX; ACT already does the full episode.

## 9. Zero-shot with median community statistics (23:05)

The MuJoCo demo Space [`multimodalart/flux-3-action-so101-sim`](https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim/commit/09edebb006d793cb215ce8c0fccfc9eacd71dfc6)
("Normalize with median new-calibration SO-101 dataset stats") overrides `policy.config.state_normalization` /
`action_normalization` with one fixed set: the median of new-calibration SO-101 datasets. That's the same hook as
our `--normalization`, so it plugs in unchanged: [`scripts/flux_convention/median_norm.json`](../scripts/flux_convention/median_norm.json).

**Does it fit our data?** Same convention as ours. Our frames normalized with each set (rest = mean of frame 0):

| | pan | lift | elbow | wflex | wroll | grip |
|---|---|---|---|---|---|---|
| Rest pose, our stats | +0.48 | −0.93 | +0.94 | −0.16 | −0.64 | −0.85 |
| Rest pose, median stats | +0.01 | −0.91 | +1.04 | +0.40 | **−1.21** | −0.85 |
| Our frames outside ±1, median stats | 0 % | 0 % | 20 % | 6 % | **85 %** | 0 % |
| Degrees per unit of predicted delta, ours / median | 0.70 / 1.68 | 3.65 / 3.36 | 3.12 / 3.23 | 1.41 / 2.13 | 0.88 / 1.70 | |

Lift, elbow and gripper look the same as with our stats. Our pan workspace becomes a narrow band (−0.36…0.14) and
pan/wrist moves are de-normalized about 2× larger. wrist_roll is the mismatch: our rest is ≈ −78°, the community's ≈ 0°.

```bash
.venv/bin/python -m rollout.server.space deploy --no-adapter --normalization scripts/flux_convention/median_norm.json \
  --revision c9e13b2aca6a0a472b3ea03fd90cff32d3e85849
```

**Offline, teacher-forced, episodes 25 and 30** (`rollout_logs/eval_mediannorm`, `eval_hold_25_30`), mean error, degrees (gripper %):

| Policy | pan | lift | elbow | wflex | wroll | **arm** | gripper |
|---|---|---|---|---|---|---|---|
| FLUX base, median stats | 5.59 | 10.50 | 8.83 | 7.57 | 4.17 | **7.33** | 3.48 |
| FLUX base, our stats (section 7.1, ep 24–30) | 3.03 | 12.39 | 11.90 | 4.85 | 1.61 | **6.76** | 2.84 |
| Hold | 1.43 | 7.92 | 5.98 | 3.70 | 0.81 | **3.97** | 1.78 |

Moving windows (14): right direction 47 %, median coverage 0.01, arm error 9.78 vs. hold 7.12. Like our stats: chance.

**Dry run** (`20261017_230748`): stopped at tick 119 by the runaway abort (elbow −64°). The commands reach out
faster than with our stats (lift −96 → −21°, elbow 93 → −39° by 3.6 s; pan to −18°), wrist_roll −92 → −18°, gripper closed.

**On the real arm** (`20261017_230816`, 20 s, safety defaults, piece on c3): 600 ticks, 19 plans, no safety stop,
the arm within 5.5° of every goal. 344 ticks clamped (pan asking down to −33°, elbow up to 103° at rest).

| Time | Joints (state) | Reading |
|---|---|---|
| 0–2.4 s | pan −5 → −14°, lift −96 → −74°, elbow 93 → 67° | heads toward c3 |
| 3.2–5.6 s | gripper 2 → 36 %, elbow down to 10°, lift −66° | reaches with the elbow, shoulder low |
| 8.8–9.6 s | **gripper 34 → 1 %** at pan −17°, then pan to +8° | closes (on nothing: the demos' grasp holds ≈ 5 %), swings sideways |
| 12–15 s | lift −96°, elbow 96° | back to rest |
| 16–19 s | pan +8° (commanded +18°), lift −57° | second reach, gripper closed, back to rest |

wrist_roll drifted −92 → +12° (toward the community rest ≈ 0°), more than with our stats.

**Reading:** the median stats are a valid alternative (same convention, same lift/elbow/gripper view) and slightly
livelier on the arm: further elbow extension (10° vs. 20°), a pick-then-move shape. But the shoulder still never
rises (max −52°; the demos grasp at about +6°), the gripper closes empty, and wrist_roll drifts further. Not better
than our own stats in any measurable way; the capability gap of section 7.3 remains.

## Safety fixes (done)

After the step-500 run, three fixes in `rollout/loop.py` (details in `docs/inference.md`, section Safety):
a runaway abort (commands > 30° outside the limits stop the episode), a stall guard that can trip under the
5°/tick cap (3° gap) and ignores the folded arm at rest, and the history recording the sent goal. Replayed on all
three real-arm logs, both bad runs stop at about 5 s, before the table contact, and the good ACT run never trips.

## Reproduce

The analysis scripts are in `scripts/flux_convention/`:

- `joint_stats.py`: section 2 (our ranges and rest pose).
- `community_groups.py`: section 3 (community_dataset_v3 groups vs. the checkpoint).
- `corpus_signs.py`: section 5 (rest pose in each training part's normalized range).
- `summarize_eval.py`: section 7.1, from `rollout run --robot dataset` log folders.
- `motion_windows.py`: section 7.2, the same log folders, scored on the windows where the demonstrator moved.

```bash
.venv/bin/python -m rollout.server.space deploy --no-adapter --normalization niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741 \
  --revision c9e13b2aca6a0a472b3ea03fd90cff32d3e85849          # or --no-normalization for the checkpoint's
.venv/bin/python -m rollout run --robot dataset --policy https://niru-5-flux3-so101-rollout.hf.space \
  --episode 24,25,26,27,28,29,30 --log-dir rollout_logs/eval_ournorm
.venv/bin/python -m rollout run --robot dataset --policy local:hold --episode 24,25,26,27,28,29,30 --log-dir rollout_logs/eval_hold
python scripts/flux_convention/summarize_eval.py rollout_logs/eval_ournorm rollout_logs/eval_hold
python scripts/flux_convention/motion_windows.py rollout_logs/eval_ournorm rollout_logs/eval_hold
.venv/bin/python -m rollout.server.space stop
```
