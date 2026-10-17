# FLUX 3 Action and our joint calibration (analysis, 2026-10-17)

**Question.** The zero-shot FLUX 3 Action SO-101 model produced nonsense for our arm, and we suspected
that our calibration (lerobot 0.6.0) differs from the one the model was trained on. Is there a mapping from
our joint values to the ones the model expects, and what is it?

**Short answer.** The difference isn't a fixed offset between calibrations. It's **normalization**. BFL
normalized every training dataset with **its own** joint range, so the model learned joint values rescaled to
about −1…1 per dataset. The released checkpoint then ships **one fixed** range (from MolmoAct-style
datasets), which puts our values far outside anything the model saw. The mapping that reproduces training
is: **normalize with our own dataset's quantiles**. It's implemented as `--normalization` and still needs to be
verified on the GPU (see [Status](#status)).

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

## Status

- Done: the analysis above, `rollout/normalization.py`, `--normalization` in the model, server, client
  and `space deploy`, and tests (18 passing).
- Deployed once (base model + our normalization, 18:43), then paused before testing.
- **Not yet verified on the GPU.** Next: teacher-forced error on the held-out episodes 24–30, base model with our
  normalization vs. with the checkpoint's (the LoRA card reports 12.49° arm / 5.82 % gripper for the base
  with the checkpoint's normalization, 6.72° / 4.24 % for LoRA step 500).
- **Before any further arm run**, the safety fixes from the step-500 run (see `docs/inference.md`) are still
  open: the stall guard can't trip with the default 5°/tick cap (6° gap > 5° cap), commands far outside the
  limits should abort rather than be clamped, and the history should record the clamped goal.

## Reproduce

The analysis scripts are in `scripts/flux_convention/`:

- `joint_stats.py`: section 2 (our ranges and rest pose).
- `community_groups.py`: section 3 (community_dataset_v3 groups vs. the checkpoint).
- `corpus_signs.py`: section 5 (rest pose in each training part's normalized range).
