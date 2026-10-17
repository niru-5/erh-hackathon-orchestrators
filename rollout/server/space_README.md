---
title: SO-101 FLUX 3 Action rollout server
emoji: 🦾
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: l40sx1
pinned: false
---

Model server for `rollout` (erh-hackathon-orchestrators, docs/inference.md): serves
[black-forest-labs/flux-3-action-so101](https://huggingface.co/black-forest-labs/flux-3-action-so101)
over HTTP for an SO-101 control loop running on another machine.

- `GET /health`, `GET /spec` (model contract), `POST /predict` (8-tick history window -> 32 absolute joint targets).
- Needs the `HF_TOKEN` secret with access to the gated model (FLUX Kommunity License).
- Client: `python -m rollout run --robot so101 --policy https://<this-space>.hf.space --task "..."`.
