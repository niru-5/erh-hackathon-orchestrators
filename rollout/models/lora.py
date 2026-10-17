"""Merge a PEFT LoRA adapter trained with LeRobot's Flux3Policy into a flux_action FluxActionPolicy.

LeRobot's adapters (e.g. Gawtam/flux3-chess-lora) name their targets ``base_model.model.dit.<module>``;
the same ``dit.<module>.weight`` tensors exist in the flux_action policy (the base checkpoint's keys).
Each LoRA pair is merged as ``W += (lora_alpha / r) * B @ A`` and the ``modules_to_save`` tensors
(the action heads) are replaced, so the policy then runs the fine-tuned model with no PEFT at inference.
"""

import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

PREFIX = "base_model.model."


def resolve_adapter(adapter: str, revision: str | None = None) -> Path:
    path = Path(adapter).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            adapter, revision=revision, allow_patterns=["adapter_config.json", "adapter_model.safetensors"]
        )
    )


def merge_lora(policy: torch.nn.Module, adapter: str, revision: str | None = None) -> dict:
    """Merge the adapter into ``policy`` in place; returns a summary. Raises on any mismatch."""
    directory = resolve_adapter(adapter, revision)
    config = json.loads((directory / "adapter_config.json").read_text())
    if (
        config.get("peft_type") != "LORA"
        or config.get("use_dora")
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
    ):
        raise ValueError("only plain LoRA adapters (no DoRA, no per-module rank/alpha) can be merged")
    rank, alpha = config["r"], config["lora_alpha"]
    scale = alpha / rank**0.5 if config.get("use_rslora") else alpha / rank
    tensors = load_file(str(directory / "adapter_model.safetensors"))
    params = policy.state_dict()  # shares storage with the parameters, so copy_ updates the model
    merged, replaced = 0, 0
    with torch.no_grad():
        for key, value in tensors.items():
            name = key.removeprefix(PREFIX)
            if ".lora_B." in name:
                continue
            if ".lora_A." in name:
                target = name.replace(".lora_A.weight", ".weight")
                up = tensors.get(key.replace(".lora_A.", ".lora_B."))
                if target not in params or up is None:
                    raise KeyError(f"adapter tensor {key} has no matching weight or lora_B")
                weight = params[target]
                delta = scale * (up.float() @ value.float())
                if delta.shape != weight.shape:
                    raise ValueError(f"{target}: LoRA delta {tuple(delta.shape)} vs weight {tuple(weight.shape)}")
                weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
                merged += 1
            else:  # modules_to_save: fully trained tensors
                if name not in params or params[name].shape != value.shape:
                    raise KeyError(f"adapter tensor {key} doesn't match a policy weight")
                params[name].copy_(value.to(params[name].dtype))
                replaced += 1
    summary = {"adapter": adapter, "revision": revision, "lora_layers": merged, "replaced": replaced, "scale": scale}
    logger.info("merged LoRA %s: %d layers (scale %.3g), %d tensors replaced", adapter, merged, scale, replaced)
    return summary
