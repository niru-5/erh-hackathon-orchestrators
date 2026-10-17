# ACT inference experiments on the SO-101

Real-arm rollouts of ACT policies trained with LeRobot on our chess dataset
(`niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741`, task "Move the figure from c3 at 6cm to D5 at 10 cm").
ACT is small enough to run on the recording laptop's CPU, so the whole loop runs locally with `local:lerobot`
(no Space, no network). How the rollout code works is in [inference.md](inference.md).

## `rtrtwrw/act_chess_c3_to_d5_15ep_h100_step4000` (2026-10-17)

ACT trained from scratch with LeRobot 0.5.1 on an H100: **15 episodes** (`[19, 12, 14, 10, 18, 15, 20, 9, 16, 28,
11, 8, 30, 29, 13]`), **4000 steps** (the final checkpoint), batch 96, lr 2e-5, other ACT defaults (resnet18 backbone).
Training loss at step 4000: 0.087. It uses all three cameras (`scene`, `wrist`, `extra`) at full resolution and
predicts 100 absolute joint targets per plan, all executed (3.3 s) before replanning.

```bash
.venv/bin/python -m rollout run --robot so101 --policy local:lerobot \
  --weights rtrtwrw/act_chess_c3_to_d5_15ep_h100_step4000 --revision 13500edcf1461296716c584d7934bc46291b4162 \
  --task "Move the figure from c3 at 6cm to D5 at 10 cm" --episode-time 20
```

### Offline, teacher-forced (episode 25, not in the training set)

Mean error vs. the recorded actions, degrees (gripper %):

| Joint | this model | do-nothing (`local:hold`) | `act_chess_c3_to_d5` (31 ep, trained on ep 25) |
|---|---|---|---|
| shoulder_pan | 3.0 | 7.0 | 1.3 |
| shoulder_lift | 23.1 | 69.3 | 7.0 |
| elbow_flex | 19.0 | 47.8 | 4.2 |
| wrist_flex | 5.2 | 10.2 | 2.6 |
| wrist_roll | 4.3 | 2.8 | 1.0 |
| gripper | 7.1 | 5.4 | 2.6 |

It clearly follows the task (lift and elbow errors a third of doing nothing), but the 100-step open-loop plans
drift in time against the demonstrator, which shows up as large lift and elbow errors.

### On the real arm: six 20 s episodes (20:30–20:38)

All with the safety defaults (demo-range limits, 5°/tick cap, stall guard, runaway abort). Every run completed the
full 20 s (600 ticks, 6 plans) with **no stall, no runaway and no bus error**. 0.43–0.47 s per plan on the CPU,
26.3–26.6 Hz, the arm moving about 88 % of the time. Logs (git-ignored): `rollout_logs/20261017_203042`, `_203144`,
`_203259`, `_203441`, `_203518`, `_203825`.

| Run | Start pose | Reach c3, gripper opens | Gripper closes | Carry pose toward d5 (pan / lift / elbow) | Release at d5 | Back at rest | Afterwards | Ticks clamped by limits |
|---|---|---|---|---|---|---|---|---|
| 1 (20:30:42) | rest | 22 % at 1.7 s | 5.2 s (3.6 % vs 2.2 % commanded) | −4° / 26° / −9° (8–10 s) | weak, 6 % | 12.8 s | second attempt, closes at 20.4 s | 49 |
| 2 (20:31:44) | rest (elbow 84°, wrist_roll −67°) | 18 % at 1.5 s | 3.8 s (6–7 % vs 2–5 %) | −4° / 4° / 20° (8–10 s) | 12 % at 14 s | never | small grasp cycles near c3 (gripper 2–12 %) | 0 |
| 3 (20:32:59) | rest | 26 % at 2.1 s | 5.4 s (3.2 % vs 3.0 %) | −4° / 11° / 13° (6.7–8 s) | 9 % at 8.8 s | 11.5 s | second attempt, closes empty at 17 s | 23 |
| 4 (20:34:41) | **lift −46°** (run 3 ended mid-air, the limp arm settled there) | only 11 % | 4.4 s (2–4 %) | −2° / 16° / 8° (5.4–6.4 s) | 12–13 % at 7.9 s | never (partial, lift −49° at 11 s) | hovered near c3, gripper 6–14 % | 0 |
| 5 (20:35:18) | rest | 28 % at 2–4 s | 5.4 s (3.8 % vs 2.5 %) | −4° / 11° / 6° (8–9 s) | **none** (stayed 3–4 %) | 12.3 s | second attempt, opens 17 %, closes at 17.8 s | 34 |
| 6 (20:38:25) | rest | 23 % at 1.4 s | 4.1 s (1.8 % vs 1.4 %) | −3° / 22° / −3° (5.4 s) | 12–14 % at 7.9 s | never (partial, lift −50° at 11.3 s) | second pick (closes 14.1 s) and carry to pan −5° (15.7–16.7 s), no clear release; heading back to c3 at 20 s | 18 |

Reading:

- **The motion is consistent.** In every run it reaches c3 (pan −10 to −14°), closes the gripper at 4–5.5 s, swings
  toward d5 (pan −2 to −4°, a longer reach), and in 5 of 6 runs opens the gripper there. In 3 of 6 it goes back to rest
  by 11.5–12.8 s and starts a second attempt; run 6 starts its second pick from a partial retreat, and runs 2 and 4 stay
  near the board. Run 6 was the fastest cycle (release at 7.9 s).
- **The release is weak**: it opens to 6–14 % at d5, against about 15 % for the 31-episode ACT, and run 5 didn't open.
- **Whether the piece is held can't be read from the joints.** While closed, the gripper sits 0.2–1.5 % above its
  command, a small gap; the 31-episode ACT held the piece at 5.5 %. Whether the piece reached d5 was judged by eye and
  not logged.
- **Start pose matters**: the arm drops when torque is released at the end of an episode, so the next run can start
  away from rest (run 4). Put the arm back at rest between runs for comparable episodes.
- **Fast moves are capped**: during the big swings ACT plans moves faster than the 5°/tick cap allows, so the arm
  lags its plan for a few ticks (shoulder_lift and elbow). The joint-limit clamps (0–49 ticks) are mostly at rest,
  where ACT asks for slightly beyond the demonstrations' range.
- **Servo glitch once**: one connection attempt before run 1 failed with servo 3 (elbow_flex) not answering ("There is
  no status packet"); the retry seconds later worked. If it persists, power-cycle the arm.

### Compared with `rtrtwrw/act_chess_c3_to_d5` (31 episodes, step 12,000)

Same architecture, same rollout path. The 31-episode model (see [inference.md](inference.md#act-fully-local-rtrtwrwact_chess_c3_to_d5-2026-10-17))
did one clean cycle in its single real-arm run (reach, grasp at 5.5 %, carry, release at 14.6 %, back to rest by
17 s). The 15-episode, 4000-step model does the same sequence, but faster (grasp at 4–5.5 s instead of 6–8 s), with a
weaker release, and it often starts a second attempt within the 20 s.

## `rtrtwrw/act_chess_c3_to_d5_10ep_step16000` (2026-10-17)

The same ACT setup (LeRobot 0.5.1, resnet18, three cameras at 480×640, 100 actions per plan, all executed) trained
from scratch on the **10 shortest episodes** (`[19, 12, 14, 10, 18, 15, 20, 9, 16, 28]`, 3,422 frames; the first 10
of the 15-episode set) for **16,000 steps** at batch 24 (the final checkpoint).

```bash
.venv/bin/python -m rollout run --robot so101 --policy local:lerobot \
  --weights rtrtwrw/act_chess_c3_to_d5_10ep_step16000 --revision 3572cef3c66048fffa0fc0d41357b6a497f64445 \
  --task "Move the figure from c3 at 6cm to D5 at 10 cm" --episode-time 20
```

### Offline, teacher-forced (episode 25, held out for both models)

| Joint | 10 ep, 16,000 steps | 15 ep, 4000 steps | do-nothing |
|---|---|---|---|
| shoulder_pan | 3.1 | 3.0 | 7.0 |
| shoulder_lift | 22.2 | 23.1 | 69.3 |
| elbow_flex | 18.3 | 19.0 | 47.8 |
| wrist_flex | 5.6 | 5.2 | 10.2 |
| wrist_roll | 4.7 | 4.3 | 2.8 |
| gripper | 7.6 | 7.1 | 5.4 |

Offline the two models are indistinguishable.

### On the real arm: three 20 s episodes (20:41–20:43)

Safety defaults. All three completed the full 20 s (600 ticks, 6 plans) with no stall, no runaway and no bus error;
0.43–0.44 s per plan, 26.5 Hz. All three started at rest. Logs: `rollout_logs/20261017_204113`, `_204211`, `_204317`.

| Run | Reach c3, gripper opens | Gripper closes (measured vs commanded) | Carry pose toward d5 (pan / lift / elbow) | Release at d5 | Back at rest | Afterwards | Ticks clamped by limits |
|---|---|---|---|---|---|---|---|
| 1 (20:41:13) | 14 % at 2.2 s | 3.8 s (3.5 % vs 3.0 %), likely empty | −3° / 5° / 19° (5.4–6.4 s), **short** | none; reopens back at c3 (pan −8°) at 7.8 s | never | three more open/close cycles near c3 (9.1, 13.3, 20.6 s) | 0 |
| 2 (20:42:11) | 24 % at 1.9 s | 5.4 s (**≈ 6 % vs 1 %**, holding) | −2° / 19° / −1 to 8° (6.4–8.9 s), **full reach, still ≈ 6 % vs 1–2 %** | not commanded; the gripper closes from 5 to 2 % as the arm lifts away at 9–10 s (the piece left the fingers) | never (heading there at 21 s) | two quick open/close cycles near c3 | 34 |
| 3 (20:43:17) | 23 % at 1.6 s | 4.6 s (1.2 % vs 1.3 %), empty | −2° / 9° / 14° (6.4 s), medium | opens 12 % midway (pan −6°) at 7.9 s | never (partial, lift −50° at 11.3 s) | deeper second reach to c3 (opens 27 %), two more open/close cycles | 3 |

Reading:

- **Reach and grasp are learned, carry and release mostly aren't.** Every run reaches c3 at 2–3 s and closes the
  gripper at 4–5.5 s. Only run 2 makes the full reach toward d5 (lift 19°, elbow −1°); runs 1 and 3 stop short and
  reopen on the c3 side or midway.
- **No deliberate release at d5 in any run.** In run 2 the piece left the fingers while the arm lifted away, without
  an open command.
- **Run 2 has the clearest grip of all ACT runs**: a 4–5 % gap between the measured and commanded gripper, held for
  about 3.5 s of carrying. The other grasps close to within 0.5 % of the command (likely empty).
- **It never returns to rest**: it cycles between c3 and a partial carry, 3–5 open/close cycles per 20 s.
- The motion is smooth and stays inside the demonstrations' range (0–34 clamped ticks).

## Comparison of the ACT models

| | `act_chess_c3_to_d5` | `…_15ep_h100_step4000` | `…_10ep_step16000` |
|---|---|---|---|
| Training episodes / steps / batch | 31 / 12,000 (of 22,000) / – | 15 / 4000 / 96 | 10 / 16,000 / 24 |
| Offline, ep 25: lift / elbow error | 7.0° / 4.2° (trained on ep 25) | 23.1° / 19.0° | 22.2° / 18.3° |
| Real-arm runs | 1 | 6 | 3 |
| Reach c3 and close the gripper | 1 of 1 | 6 of 6 | 3 of 3 |
| Full reach toward d5 | 1 of 1 | 5 of 6 (lift 11–26°; run 2 short, 4°) | 1 of 3 (lift 19°; others 5° and 9°) |
| Gripper opens at d5 | 1 of 1 (14.6 %) | 5 of 6 (6–14 %) | 0 of 3 |
| Back at rest within 20 s | 1 of 1 (17 s) | 3 of 6 (11.5–12.8 s) | 0 of 3 |
| Clear grip (measured ≫ commanded) | yes (5.5 %) | at most 1.5 % gap | 1 of 3 (run 2, ≈ 5 % gap) |
| Safety stops | none | none | none |

- **More episodes beat more steps.** The 10-episode model trained 4× longer matches the 15-episode model offline but
  does less of the task on the arm: it rarely carries to d5 and never releases there. The 15-episode model does the
  whole sequence most of the time, with a weak release. The 31-episode model did one clean cycle.
- **Offline error doesn't rank these models**: 10 ep and 15 ep are within 1° offline on episode 25 but behave clearly
  differently on the arm. Real-arm runs (several per model, from rest) are the comparison that matters.
- **Whether the piece reached d5 was judged by eye and isn't logged.** The gripper's measured-vs-commanded gap is the
  only signal in the logs, and it's weak.
- All nine runs of the two new models were safe: no stall, runaway or servo error during a run (one failed
  connection attempt before the first 15-episode run, fixed by retrying).
