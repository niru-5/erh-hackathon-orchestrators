# erh-hackathon-orchestrators

## Docs

- [SO-101 + chess simulation, RL and real fine-tuning](docs/simulation.md): plan for a MuJoCo (then
  Isaac Sim) world with two cameras (wrist + side), RL in sim, fine-tuning on real data, final rollout.
- [Simulation TODO: task v1](docs/simulation_todo.md): step-by-step checklist (MuJoCo setup, SO-101 import,
  real setup measurements, calibration, pieces, cameras, first pick-and-place, sim-to-real check).

- [SO-101 data collection (LeRobotDataset v3.0)](docs/data_collection.md): record teleoperated
  demos with `scripts/record_so101.sh --task "<prompt>" --episodes <N>`.
- [SO-101 policy rollout (FLUX 3 Action, local or remote)](docs/inference.md): run a pretrained policy on
  the arm with the model on a remote GPU: `python -m rollout run --robot so101 --policy <server URL> --task "<prompt>"`.
