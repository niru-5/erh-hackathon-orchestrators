# Report: a chess-playing SO-101 arm, from one move to many

Team Orchestrators, 2026-10-17 to 2026-10-18

## Summary

We set out to build a robot that plays chess on a real board, so that friends far apart can play each other on a
physical board. We started with the smallest useful version of the problem: can a low-cost SO-101 arm move one piece
from one square to another?

- **One move (c3 → d5), 31 demos.** An ACT policy trained on them did the full pick-and-place on the real arm.
- **FLUX 3 Action, zero-shot and fine-tuned, first failed badly:** it drove the arm into the table and sent runaway
  commands. The cause was normalization. The checkpoint's normalization parameters come from a different SO-100
  dataset with a different joint convention, and they aren't documented.
- **Fine-tuning FLUX with our own normalization fixed it.** Offline error dropped from 13.4° to 2.8°, and the best
  checkpoint did a complete pick-and-place in one episode.
- **We scaled up data collection** to moves along the b, c and d files (b1 → b2 … b7, and the same for c and d). A
  camera problem forced us to recollect part of it.
- **Super ACT** puts the task text into ACT's 6-dimensional input so that one ACT model can learn many different
  moves.
- **FLUX fine-tuned on the multi-move data** works in distribution and seems to generalize to other parts of the
  board. We checked what it "thinks" by decoding its predicted future video frames and by looking at its attention
  maps.

**Conclusion:** Super ACT is a cheap, practical way to train offline. Fine-tuned FLUX is the better choice for
generalizing to new data. If FLUX gets strong enough, it can generate the data to train many small, fast ACT
policies.

## 1. Motivation

- Chess is fun, easy to judge (the piece is on the right square or it isn't), and a natural way to stay in touch
  with friends who live far away: their online moves appear on your real board.
- Under the game, the task is a precise pick-and-place of small, similar objects onto a grid (squares are a few
  cm wide), following a language instruction such as "move the figure from c3 to d5". The same skill applies to
  many real tasks beyond chess.

## 2. Setup

| Part | What we used |
|---|---|
| Arms | Two SO-101 arms: the leader is moved by hand, the follower copies it (teleoperation) |
| Cameras | `scene` (fixed top view of the board), `wrist` (on the gripper), `extra` (side view, recorded but not used for training) |
| Recording | Official `lerobot-record` (LeRobot 0.6.0, LeRobotDataset v3.0) at 30 Hz, wrapped in `scripts/record_so101.sh` |
| Compute | Laptop with a CPU only (no NVIDIA GPU). ACT runs on it locally. FLUX (about 25 GB of weights) runs on an A100 in a Hugging Face Docker Space |
| Rollout | `python -m rollout run`: the laptop sends observations, the server returns a plan of actions, and a safety layer checks every command |

Getting a true 30 Hz took work. The Rerun live viewer was silently slowing recording to 0.2–7.9 Hz. With the
viewer off and V4L2 + MJPG cameras, recording runs at 29.9 Hz with 0 late frames. A LeRobot dataset can't prove its
own frame rate (`timestamp = frame_index / fps`), so we time every frame against the real clock
(`scripts/record_timed.py`). Details: [docs/data_collection.md](docs/data_collection.md).

## 3. Phase 1: one move, c3 → d5

### Data

`niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741`: 31 episodes, 13,229 frames (7 min 21 s), with the prompt
"Move the figure from c3 at 6cm to D5 at 10 cm".

### ACT

ACT trained on the 31 episodes (`rtrtwrw/act_chess_c3_to_d5`) ran on the laptop CPU at about 0.45 s per
100-action plan. On its **first real-arm episode** it did the full sequence: reach → grasp → carry → release on d5 →
return to rest. We also compared smaller runs: 15 episodes / 4000 steps and 10 episodes / 16,000 steps. More
episodes helped more than more training steps. Offline error didn't separate the models, but the real arm did.
Details: [docs/act_inference_experiments.md](docs/act_inference_experiments.md).

## 4. Phase 2: FLUX 3 Action, and the normalization problem

### What went wrong

While ACT trained, we tried FLUX 3 Action zero-shot and started a LoRA fine-tune on the same 31 episodes.

- **Zero-shot** didn't do the task. Offline it looked "389° off" on shoulder_lift.
- **The fine-tuned LoRA was worse on the real arm.** Step 250 pushed into the table, and the servos shut down
  from overload (the follower needed a power cycle). Step 500 ran away and commanded shoulder_lift to 844° and the
  elbow to 374°.

### Diagnosis

We compared our joint ranges with those of the SO-101/SO-100 data FLUX was trained on
([docs/flux_joint_convention.md](docs/flux_joint_convention.md)):

- FLUX predicts per-tick changes and normalizes joints with q01/q99 quantiles. The shipped parameters exactly match
  one training dataset (`VoicAndrei/so100_kitchen`, old SO-100 calibration: shoulder_lift 85…184°). Ours is
  −97…39°.
- So our arm's rest pose normalized to about −2.8 to −4.7 on a scale meant to be −1…1. The model was seeing
  states it had never seen. The per-tick ranges were also 4–5× ours on pan, wrist_flex and wrist_roll.
- None of this was documented. We found it by comparing the joint statistics directly.
- BFL normalized **each training dataset with its own quantiles**, which pointed to the fix.

Part of the zero-shot "389°" was our own evaluation bug: replaying a recording while feeding the model its own
commands as history makes errors compound. We switched to **teacher-forced** offline evaluation.

### Fix: fine-tune and serve with our own normalization

| Policy | Offline arm error | Real arm |
|---|---|---|
| Base FLUX, checkpoint normalization | 13.36° | — |
| Base FLUX, our normalization | 6.76° | safe, turns toward c3 and opens the gripper, one empty close |
| LoRA trained with our normalization, step 500 | 5.3–5.6° | aimed reaches, gripper open at the piece |
| LoRA trained with our normalization, step 2500 | 2.8–3.1° | reach, grasp, carry, release (split over three episodes) |
| **LoRA trained with our normalization, step 4500** | **2.83°; 87 % right direction while moving** | **complete pick-and-place in one episode, twice** |

(Repeating the last command scores about 4.0° over a whole episode and 7.1° on the moving parts. Whole-episode
error rewards standing still, so we rank checkpoints on the moving parts.)

This showed that **FLUX can be used** on our arm, as long as it is trained and served with the normalization of the
data it sees.

### A safety layer for learned policies

The table hit and the runaway led us to put a safety layer between the policy and the motors: joint limits taken
from the demos, a 5° per-tick cap, a stall guard, and a runaway abort. The history fed back to the model records the
command actually sent, so a clamped command can't snowball. Replaying the earlier failures with the new guards
stops each one within about 5 s, and good runs are unaffected. Details: [docs/inference.md](docs/inference.md).

## 5. Phase 3: scaling up data collection

To go from one move to many, we recorded moves along the files, with one source square on rank 1 moving to every
other square in its file:

- b1 → b2, b1 → b3, … b1 → b7
- c1 → c2 … c7
- d1 → d2 … d7

Each move is its own dataset of about 6–8 episodes, pushed to the Hugging Face Hub (`niru-5/chess_dataset_*`).

**Camera problem and recollection.** Partway through, the scene camera moved. Comparing the first frames of each
dataset with image registration showed 90–245 px shifts and 4–7° of rotation in the older series, so we recollected
the data (the `chess_dataset_new_*` series). In the new series, all 21 datasets agree to within 0.8 px and 0.05°.
The old and new series shouldn't be mixed in training without accounting for the camera change.

## 6. Phase 4: Super ACT, one ACT model for many moves

Plain ACT ignores the task prompt, so one ACT model learns only one move. **Super ACT** puts the task text into
the 6-dimensional vector ACT uses as input, so a single model learns the different movements and the prompt
chooses which one to do.

- It trains and runs cheaply (ACT runs on the laptop CPU).
- It works well for the moves it was trained on, so it is a practical offline training option.

## 7. Phase 5: FLUX fine-tuned on the multi-move data

We fine-tuned FLUX on the multi-move data (`Gawtam/flux3-chess-lora-28moves`, served with its own normalization).

- **In distribution** (moves it was trained on), it works.
- It **seems to generalize** to other parts of the board, which is FLUX's main advantage over ACT.
- In the logged debug runs (b1 → b4, a1 → a4), reaches to the piece were well aimed. The remaining failure was
  grasping: the gripper closed just off or above the piece. The fingers stopped at shoulder_lift about −8°, where
  the demos go to +3…+12°.
- Joint limits have to come from the move's own dataset (`--limits-from`): b1 needs pan −24° and a1 needs −27°,
  while the c3 → d5 limits stop at −18°.

## 8. Is the model "thinking" correctly?

Low joint error doesn't mean the model understands the scene, so we looked inside it in two ways.

### Predicted future frames

With each action chunk, FLUX also predicts about 1.4 s of future video (scene and wrist views), and the standard
sampler throws it away. We decode it with the VAE that ships with the policy. `rollout run --debug` saves it with
every frame and plan, and `python -m rollout view` builds a viewer that shows imagined and real frames side by
side. In the multi-move runs, **the imagined frames matched what the arm actually did**. The model's idea of where
it was going was right even when the grasp missed.

### Attention maps

We also looked at the attention maps to check that the model attends to the right places: the piece, the source
and target squares, and the gripper.

## 9. Conclusions

1. **Normalization matters more than model size.** A strong foundation model failed until its inputs matched its
   training data. Each dataset needs its own normalization, and a LoRA must be served with the normalization it was
   trained with.
2. **Super ACT is a cheap way to train offline.** It is fast to train, runs on a CPU, and handles a known set of
   moves with one model.
3. **Fine-tuned FLUX is the way to generalize** to new moves and new data.
4. **The two can work together.** If FLUX is strong enough, it can run new tasks and generate demonstrations to
   train many small, fast ACT policies (distillation), getting FLUX's generalization at ACT's cost.
5. **Test on the real arm early.** Offline numbers can mislead (a model that holds still scores well), and they
   didn't separate models that behaved very differently on the arm.

## 10. Open problems and next steps

- **Stop-and-go execution:** the arm waits 67 % of the time for the next FLUX plan. Asynchronous replanning
  (computing the next plan while the current one runs) would fix this.
- **Grasp height and wrist alignment** for FLUX on new moves.
- **A contact-aware stall guard:** the current one stops runs at legitimate contact (fingers on the piece).
- **Placement checking from the camera** rather than gripper readings.
- **A full game:** a chess engine, board-state reading from the scene camera, captures, and remote play between
  friends.

## References

- [docs/worklog.md](docs/worklog.md): full timeline, debugging lessons and scoreboard
- [docs/data_collection.md](docs/data_collection.md): recording pipeline and datasets
- [docs/inference.md](docs/inference.md): rollout architecture, safety layer, and all 61 real-arm runs
- [docs/flux_joint_convention.md](docs/flux_joint_convention.md): the normalization analysis
- [docs/act_inference_experiments.md](docs/act_inference_experiments.md): ACT runs compared
