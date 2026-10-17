# SO-101 policy rollout (FLUX 3 Action, local or remote)

This doc covers running a pretrained policy on the SO-101 **before any fine-tuning**: a one-shot
test of [`black-forest-labs/flux-3-action-so101`](https://huggingface.co/black-forest-labs/flux-3-action-so101)
on our rig. The model needs a large GPU (about 32 GB in BF16), and the recording laptop has none, so the
model runs on a remote GPU (a Hugging Face Space or any GPU box). The laptop drives the arm and sends
observations over HTTP. The code is in [`rollout/`](../rollout).

The same code can also run the model in-process on a machine with a GPU, or use a mock model (no GPU)
to test the robot side.

## The model's contract

From the model card, `config.json` and the reference code in
[black-forest-labs/flux-action](https://github.com/black-forest-labs/flux-action) (commit `830f1ce`,
`flux_action/inference/so101.py`, `policy.py`, `processing/history.py`):

| | |
|---|---|
| Cameras | `scene` then `wrist`, any size; each is resized to 256×256 and placed side by side on a 256×512 canvas (`camera_layout: side_by_side`) |
| State | 6 joint positions in our dataset's units: degrees for the 5 body joints, 0–100 % for the gripper (`shoulder_pan … gripper`) |
| Task | Free-text instruction (Qwen3-VL-4B text encoder) |
| History | `n_obs_steps = 8` ticks of state **and** of the commands issued before each tick. Images are used only at window positions 0 and 7 (`history_snapshots = 2`) |
| Output | 42 actions per plan (`chunk_size`). The first 32 are executed at 30 Hz (`n_action_steps`, 1.07 s), then it replans |
| Action space | The model predicts per-tick joint **deltas** (absolute gripper). The reference code adds them onto the last issued command, so the output is **absolute joint targets** in the same units as `observation.state`, the same as our `action` column |
| Sampling | 4 Euler steps, shift 6.93, guidance 3, seed 42, BF16 |
| Hardware | NVIDIA GPU, about 32 GB for BF16 inference (BFL docs). torch 2.10 + CUDA 12.8, transformers 5.16.1, NATTEN 0.21.6 for the video VAE |
| License | FLUX Kommunity License (gated repo, auto-approved: accept it once on the model page with the account whose token the server uses) |

Our dataset already matches this contract: same camera names, same joint order and units.

### Why one request per plan is enough

The reference control loop (`FluxActionPolicy.select_action`) is called every tick. It:

1. appends the tick's observation (state, images) and the **previously issued command** to an 8-tick
   history (on the first tick of an episode the history is filled with 8 copies, and the command is the
   measured state);
2. if the action queue is empty, predicts a plan from that window and queues 32 steps;
3. pops one step and adds it onto the previous command (gripper taken as is). That sum is the next command.

Steps that are already queued never look at new observations. A plan is therefore fixed the moment it
is made, and it equals `predict_action_chunk(window)[:, :32]` on the explicit 8-tick window
(cumulative sum of the deltas from the window's last command). New observations only matter for the next
plan, through the history.

So the robot side can keep the history itself, and send it **once per plan** (every 32 ticks) to a
**stateless** model server. A round trip every tick (33 ms) would be impossible over the internet; one
per 1.07 s is workable. `rollout/tests/test_flux_equivalence.py` checks against the real `flux_action`
code (tiny model on CPU) that this gives exactly the same actions as calling `select_action` every tick.

## Architecture

```
 recording laptop (no GPU)                                       GPU host (HF Space / GPU box)
 ┌──────────────────────────────────────────────────┐            ┌────────────────────────────────┐
 │ rollout.cli  (python -m rollout run ...)          │            │ rollout.server (FastAPI)       │
 │                                                   │            │                                │
 │  Robot ──observe()──► control loop ──► History    │            │  GET  /spec    model contract  │
 │  (SO-101 via lerobot,   │  30 Hz       (8 ticks)   │   HTTPS    │  POST /predict window -> plan  │
 │   or a dataset replay)  │                │         │  JSON +    │        │                       │
 │        ▲                │ queue empty?   ▼         │  JPEG      │        ▼                       │
 │        │                ├──────────► PolicyBackend ├───────────►│     Model                      │
 │        │                │            ├ RemoteBackend│◄───────────┤     ├ FluxModel (flux_action) │
 │  act(action) ◄── queue ◄┘            └ LocalBackend │  32 actions│     └ HoldModel (mock)        │
 │  (+ safety limits)                     (in-process) │            │                                │
 └──────────────────────────────────────────────────┘            └────────────────────────────────┘
```

Three layers, each one swappable on its own:

| Layer | Interface | Implementations |
|---|---|---|
| **Robot** (`rollout/robots/`): where observations come from and where actions go | `observe() -> Observation(state, images)`, `act(action)` | `SO101Robot`: the real follower arm and cameras via lerobot, same ports/cameras/calibration as `scripts/record_so101.sh`. `DatasetRobot`: replays a recorded episode (open-loop evaluation, no hardware) |
| **Backend** (`rollout/backends/`): how the robot side reaches a model | `spec() -> PolicySpec`, `predict(Window) -> Plan` | `RemoteBackend`: HTTP to `rollout.server` (on an HF Space, a GPU box, or `localhost`). `LocalBackend`: calls a model in the same process |
| **Model** (`rollout/models/`): turns a window into a plan, runs where the GPU is | `spec()`, `predict(Window) -> Plan` | `FluxModel`: FLUX 3 Action SO-101 via `flux_action`. `HoldModel`: returns "stay where you are" (tests the whole pipeline without a GPU) |

`RemoteBackend` and `LocalBackend` give the robot side the same interface. The server wraps any model
with the same `Window`/`Plan` types, so a model behaves the same in-process or remote.

The model owns the contract: on connect the robot side fetches `PolicySpec` (`camera_keys`,
`n_obs_steps`, snapshot indices, `n_action_steps`, `fps`, image size) and sizes its history from it.
Nothing about FLUX is hard-coded on the robot side.

### Control loop (sync replanning)

```
reset: history.clear(); queue.clear(); last_command = None
every tick (30 Hz):
    obs = robot.observe()                         # state (6) + scene/wrist images
    history.append(obs, last_command)             # same rule as flux's ObservationHistory
    if queue is empty:
        plan = backend.predict(history.window(task))  # robot holds position while waiting
        queue = plan.actions[:n_action_steps]
    action = queue.pop()
    robot.act(safety(action))
    last_command = action                         # the model's command, before safety clipping
    sleep until the next tick
```

While a request is in flight the arm holds its last command, so motion comes in bursts: 1.07 s of
motion, then a pause of one round trip. This keeps the model's inputs exactly as in its reference
loop. Overlapping the next request with execution would hide the pause, but then the window would
be older than the model expects. That's a possible next step (see [Next steps](#next-steps)).

### Wire protocol (HTTP + JSON)

`GET /spec` returns the model contract:

```json
{"name": "flux-3-action-so101", "camera_keys": ["scene", "wrist"], "n_obs_steps": 8,
 "snapshot_indices": [0, 7], "chunk_size": 42, "n_action_steps": 32, "fps": 30.0,
 "image_hw": [256, 256], "action_names": ["shoulder_pan.pos", "...", "gripper.pos"]}
```

`POST /predict` takes one window and returns one plan:

```json
// request
{"task": "Move the figure from c3 at 6cm to D5 at 10 cm",
 "states":   [[...6 floats...] x 8],          // measured states, oldest first
 "commands": [[...6 floats...] x 8],          // command issued before each state
 "images": {"scene": {"0": "<base64 JPEG>", "7": "<base64 JPEG>"},
            "wrist": {"0": "<base64 JPEG>", "7": "<base64 JPEG>"}}}
// response
{"actions": [[...6 floats...] x 32],          // absolute joint targets, degrees / gripper %
 "timing": {"decode_ms": 3.1, "inference_ms": 412.0, "total_ms": 416.0}, "model": "flux-3-action-so101"}
```

- Only the snapshot frames (window positions 0 and 7) are sent: 4 JPEGs per plan. The other 6 window
  positions are filled with the nearest snapshot on the server; the model never reads them (it encodes
  only `canvas[:, index]` for the snapshot indices).
- Images are resized on the robot side to the model's `image_hw` (256×256, `INTER_AREA`) and sent as
  JPEG (quality 90). With our cameras a request is about 90 KB (measured). `--full-res-images` sends the camera frames as
  captured.
- Auth: a private HF Space needs `Authorization: Bearer <HF token>`; the client sends the token from
  `hf auth login` (or `HF_TOKEN`). A self-hosted server can require `X-Rollout-Key` (env `ROLLOUT_API_KEY`).

### Where the model runs

| Target | How | Notes |
|---|---|---|
| **Hugging Face Space (Docker, GPU)** | `python -m rollout.server.space deploy` creates a private Docker Space from `rollout/server/Dockerfile` on an `a100-large` (80 GB) GPU | $2.50/h while starting or running, run only during sessions (see [Sessions and budget](#sessions-and-budget)). Needs secret `HF_TOKEN` with access to the gated model |
| Any GPU box | `docker build -f rollout/server/Dockerfile .` and run it, or `python -m rollout.server --model flux` in a flux-action environment | Point the client at `http://<host>:7860` |
| In-process | `--policy local:flux` | Only on a machine with a ≥ 32 GB GPU |
| Mock | `--policy local:hold`, or a server started with `--model hold` | No GPU. Tests the robot loop, the HTTP path and timing |

An HF Inference Endpoint with a custom container works too: it is the same Docker image (port 7860).
"Inference Providers" (the serverless API) don't serve this model, and ZeroGPU Spaces don't fit
a 30 Hz control loop (the GPU is allocated per call, with queueing).

### Sessions and budget

The Space only runs while we work with it. Total spend is capped at **$20**, about 8 h of A100 ($2.50/h).
We use `a100-large` because Hugging Face had no free `l40sx1` (48 GB, $1.80/h) on 2026-10-17 ("Scheduling
failure: not enough hardware capacity"). `start --hardware l40sx1` switches back when there is capacity.

| When | Command | Effect |
|---|---|---|
| Once | `python -m rollout.server.space deploy` | Creates the private Space `niru-5/flux3-so101-rollout`, uploads the server and Dockerfile, stores the `HF_TOKEN` secret, requests the GPU, waits until it serves |
| Session start | `python -m rollout.server.space start` | Resumes the paused Space (switching to `--hardware` if it differs) and waits until `RUNNING` |
| Any time | `python -m rollout.server.space status` | Stage, URL, estimated spend vs. budget |
| Session end | `python -m rollout.server.space stop` | Pauses the Space. Billing stops, and only `start` (the owner) can resume it |

- Hugging Face bills per minute **only while the Space is starting or running**. Building, sleeping and
  paused time are free ([docs](https://huggingface.co/docs/hub/spaces-gpus#billing)).
- `start`/`deploy` log a session start in `~/.cache/so101-rollout/space_usage.json`, `stop` logs its end.
  The spend estimate (hours × the GPU's hourly price) counts start-up time too, so it errs high. `start`/`deploy` refuse
  when less than 15 min of budget is left (`--budget` overrides).
- Backstop: the Space also sleeps after 15 min without requests (`--sleep-time`), so a session that ends
  without `stop` stops billing on its own. A sleeping Space wakes on the next request, so use `stop`
  (pause), which can't be woken by accident.
- Each start downloads the weights again (≈ 25 GB: policy + text encoder + video VAE). That's billed
  start-up time, a few minutes per session.

### Latency budget

Per plan, the arm pauses for: upload (≈ 90 KB) + network round trip + server decode + model inference.
Measured on localhost with the mock model: HTTP + JPEG decode add about 5 ms over the model's own time.
BFL reports about 80 ms per plan on an H200 for a compiled 512×512 model. Our A100, eager BF16, will be slower.
The client prints and logs every request's `total_ms` (client-measured) and the server's `inference_ms`,
so the split between network and GPU is visible. Rough guide: with a 500 ms round trip the arm moves
1.07 s, pauses 0.5 s, so about 68 % duty.

### Safety

The model card: *nothing in the model bounds joint velocity, force or workspace*. The robot side:

- `--max-relative-target` (default 10°/tick): lerobot clips each goal to within this of the present position.
- `--limits-from <dataset>`: clamps every joint to the range seen in a dataset (`meta/stats.json`
  min/max ± `--limit-margin`, default 10°). Recommended: our main dataset.
- `--dry-run`: run everything (cameras, model, timing) but never send goals. The arm holds still.
- Esc stops; → ends the episode early. On exit (also Ctrl+C or an error) lerobot disables torque
  (`disable_torque_on_disconnect`), so **the arm goes limp and drops**: keep a hand under it, and keep a hand near
  the power switch.
- While a plan request is in flight the arm holds its last goal (torque on).
- Episodes stop after `--episode-time` (default 30 s).

## Using it

All commands run from the repo root with the lerobot venv (`.venv`).

```bash
# 1. no GPU, no robot: mock model, replay a recorded episode (tests the whole loop)
.venv/bin/python -m rollout run --robot dataset --policy local:hold \
  --dataset niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741 --episode 0

# 2. same, through HTTP: start a mock server, point the client at it
.venv/bin/python -m rollout.server --model hold --port 7860 &
.venv/bin/python -m rollout run --robot dataset --policy http://localhost:7860 --episode 0

# 3. deploy the real model to a private GPU Space once (paid), then start/stop it per session
.venv/bin/python -m rollout.server.space deploy
.venv/bin/python -m rollout.server.space start    # session start
.venv/bin/python -m rollout.server.space stop     # session end

# 4. open-loop check of FLUX on a recorded episode: predicted vs. recorded actions
.venv/bin/python -m rollout run --robot dataset --policy https://niru-5-flux3-so101-rollout.hf.space \
  --dataset niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741 --episode 0

# 5. real arm, no motion first, then with motion
.venv/bin/python -m rollout run --robot so101 --policy https://niru-5-flux3-so101-rollout.hf.space \
  --task "Move the figure from c3 at 6cm to D5 at 10 cm" --dry-run
.venv/bin/python -m rollout run --robot so101 --policy https://niru-5-flux3-so101-rollout.hf.space \
  --task "Move the figure from c3 at 6cm to D5 at 10 cm" \
  --limits-from niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741 --episodes 3
```

Each run writes `rollout_logs/<timestamp>/` (git-ignored): `ticks.jsonl` (per tick: time, state,
command, sent goal, plan id), `plans.jsonl` (per request: latency, server timing, bytes) and
`summary.json`. On the dataset robot the summary also has the per-joint error between predicted and
recorded actions.

On the dataset robot the observations come from the recording, not from the policy's own actions
(open loop), so the error measures how close the model's commands stay to the demonstrator's. It doesn't
show whether the model would complete the task. Compare against the do-nothing baseline, `--policy local:hold`, on the
same episodes. Its error is what "never moving" scores. For example, episode 19 with `local:hold`: shoulder_lift
64°, elbow_flex 50° mean error over the episode; 5° / 3° over the first plan. Wall time and Hz are not
meaningful on the dataset robot (it doesn't run in real time; video decoding sets the pace).

## First results (2026-10-17, A100 Space, open loop on episode 0)

The full path works with the real model: recorded episode 0 of our main dataset → `RemoteBackend` → HTTPS →
Space (`a100-large`) → FLUX 3 Action → 15 plans of 32 actions, no errors.

| | Value |
|---|---|
| Server inference per plan | 1.07 s (p50; eager BF16, no `torch.compile`) |
| Round trip per plan (client) | 1.54 s p50, 2.25 s p95, 2.75 s max. So about 0.45 s is upload + network |
| On the arm this would mean | 1.07 s of motion, then about 1.5 s holding still (≈ 40 % duty) |
| Space start (paused → serving) | about 4.5 min (image start, weight download, two warm-up predictions) |

**The predicted actions are far off, because of a joint-convention mismatch, not (only) model quality.**
Mean error vs. the recorded actions: shoulder_lift 389°, elbow_flex 196°, wrist_flex 84°, wrist_roll 71°,
shoulder_pan 54°, gripper 8.5 %. Even the first plan is off by 63° (lift) and 67° (roll). The model's own
normalization quantiles (`policy_preprocessor_step_3_*.safetensors`, `dataset_statistics.json`) show the
state range it was trained on:

| Joint | Model training state, q01 … q99 | Our dataset state, q01 … q99 |
|---|---|---|
| shoulder_pan | −92.0 … 54.1 | −13.2 … 2.1 |
| shoulder_lift | **84.8 … 184.2** | −95.5 … 23.8 |
| elbow_flex | **61.4 … 178.9** | −4.4 … 94.5 |
| wrist_flex | −24.0 … 80.6 | 64.6 … 92.6 |
| wrist_roll | −124.5 … 60.2 | −85.3 … −65.5 |
| gripper | 0.5 … 53.6 | 2.1 … 22.7 |

Our calibration (lerobot 0.6.0, `use_degrees=true`) puts 0° at the middle of each joint's calibrated range.
The SO-101 episodes in `lerobot/community_dataset_v3` that the model learned from use other zeros:
shoulder_lift looks like ours + ~180°, and elbow_flex and the wrist joints are shifted too, possibly with a
flipped sign. Fed our raw state, the model sees an arm pose it never saw, and its delta predictions explode.

**Don't run this model on the arm until that's mapped.** It would command joints far outside our range;
`--limits-from` would clamp them, but the motion would still be meaningless. Next step: estimate a per-joint
sign and offset between the two conventions (e.g. compare the rest pose at the start of community episodes with ours,
and confirm with a known pose on the arm). Then apply it in the client, before `History` and after the plan, so the
model sees its training convention and the arm gets ours.

## Testing

```bash
PYTHONPATH= .venv/bin/python -m pytest rollout/tests
```

`PYTHONPATH=` keeps ROS's pytest plugin out: the shell profile adds `/opt/ros/jazzy` to the path, and that
plugin fails to import in our venv.

- `test_history_codec.py`: history rules (first-tick padding, command bookkeeping, snapshots, resizing) and
  the JSON/JPEG round trip.
- `test_server_loop.py`: a real HTTP server with the mock model (spec, predict, API key, bad input → 422);
  the control loop replans every 32 ticks and chains commands across plans; safety clamping doesn't
  feed back into the model's command history; joint limits from `stats.json`.
- `test_flux_equivalence.py`: BFL's real `FluxActionPolicy` code with a tiny random network on CPU. 80
  ticks through flux's own `select_action` loop vs. `History` + `FluxModel.predict` once per plan: the
  actions match to 1e-4. It also fails if the wrong snapshot frames are sent (checked by breaking it on
  purpose). It needs `flux_action` on the path (skipped otherwise), see the file's docstring.

## Code layout

```
rollout/
├── __main__.py, cli.py     # python -m rollout run ...
├── contract.py             # PolicySpec, Observation, Window, Plan (numpy only)
├── history.py              # 8-tick history, mirrors flux's ObservationHistory
├── codec.py                # JSON + JPEG wire format
├── loop.py                 # control loop, safety limits, logging
├── backends/               # local.py (in-process), remote.py (HTTP)
├── models/                 # flux.py (FLUX 3 Action), hold.py (mock)
├── robots/                 # so101.py (lerobot follower + cameras), dataset.py (replay)
├── server/                 # app.py (FastAPI), Dockerfile, requirements.txt, space.py (deploy/start/stop)
└── tests/                  # history/codec/server/loop tests, flux equivalence test
```

## Next steps

- **Overlap requests with execution** (async inference): send the next window about one round trip before the
  queue runs out. This hides the pause, but the window is then a few ticks old.
- **Fine-tune** with the model's `lora.json` recipe on our dataset (`lerobot-train` needs `lerobot.policies.flux3`, which
  our lerobot 0.6.0 doesn't have yet; BFL's [SO-101 LoRA guide](https://github.com/black-forest-labs/flux-action/blob/main/docs/so101-lora.md)
  uses its own environment), then serve the fine-tuned weights with `--weights`.
