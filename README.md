# erh-hackathon-orchestrators

## Docs

- [SO-101 data collection (LeRobotDataset v3.0)](docs/data_collection.md): record teleoperated
  demos with `scripts/record_so101.sh --task "<prompt>" --episodes <N>`.
- [SO-101 policy rollout (FLUX 3 Action, local or remote)](docs/inference.md): run a pretrained policy on
  the arm with the model on a remote GPU: `python -m rollout run --robot so101 --policy <server URL> --task "<prompt>"`.
