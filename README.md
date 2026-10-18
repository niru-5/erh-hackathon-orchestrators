# erh-hackathon-orchestrators

Goal: Exploring world action model for moving chess pieces. 

How did we start -> 
Lets take a simple task -> Moving a horse piece from C3 to D5 -> 31 episodes. 
What did we do? 
-> we trained a finetuned flux model -> did not work, normalization issue. 
-> we tried the one shot 



## Docs

- [Worklog](docs/worklog.md): what was tried, what broke and how it was fixed, 2026-10-17 to 2026-10-18, with a
  scoreboard of every policy run on the arm.

- [SO-101 data collection (LeRobotDataset v3.0)](docs/data_collection.md): record teleoperated
  demos with `scripts/record_so101.sh --task "<prompt>" --episodes <N>`.
- [SO-101 policy rollout (FLUX 3 Action, local or remote)](docs/inference.md): run a pretrained policy on
  the arm with the model on a remote GPU: `python -m rollout run --robot so101 --policy <server URL> --task "<prompt>"`.
- [ACT inference experiments](docs/act_inference_experiments.md): ACT policies run locally on the laptop CPU,
  real-arm runs compared (15 episodes / 4000 steps: six 20 s episodes; 10 episodes / 16,000 steps: three).
