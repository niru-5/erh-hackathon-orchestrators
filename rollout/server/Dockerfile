# FLUX 3 Action SO-101 model server (rollout.server) for a GPU host: a Hugging Face Docker Space,
# an Inference Endpoint (custom container) or any machine with an NVIDIA GPU (~32 GB).
#
# Build from the repo root:  docker build -f rollout/server/Dockerfile -t so101-rollout-server .
# Run:                       docker run --gpus all -p 7860:7860 -e HF_TOKEN=hf_... so101-rollout-server
#
# The torch wheels bundle CUDA 12.8; the host only needs an NVIDIA driver that supports it.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
COPY rollout/server/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt -f https://whl.natten.org/

# Hugging Face Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
COPY --chown=user rollout/ rollout/

EXPOSE 7860
# Weights (~14 GB policy + shared encoders) download at start; with Space persistent storage they are
# cached in /data across restarts. HF_TOKEN must have access to the gated model.
CMD ["sh", "-c", "if [ -w /data ]; then export HF_HOME=/data/huggingface; fi; exec python -m rollout.server --model flux --port ${PORT:-7860}"]
