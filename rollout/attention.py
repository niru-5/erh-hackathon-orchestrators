"""run --debug: where a policy's action tokens look, as ``Plan.debug["attention"]`` (shown by the viewer).

    {"maps": {camera: {window position: [[...]]}},  # one value per image token of that input image, relative:
                                                    # 1 = the plan's most-attended image token over all cameras
     "shares": {"scene": 0.23, "task text": 0.05, ...},  # of the action tokens' attention, summing to 1: each
                                                         # camera (all its positions), then the other token groups
     "about": "..."}                                # which tokens and layers were averaged

Each model fills it from its own token layout (rollout/models/flux.py, rollout/models/lerobot_policy.py).
"""

import numpy as np


def summary(images: dict[str, dict[int, np.ndarray]], other: dict[str, float], about: str) -> dict:
    """``images``: camera -> window position -> (h, w) attention weights of that image's tokens; ``other``: the
    attention on every other token group. Weights are fractions of the action tokens' attention (rows of softmax)."""
    peak = max((float(grid.max()) for frames in images.values() for grid in frames.values()), default=0.0) or 1.0
    shares = {camera: float(sum(grid.sum() for grid in frames.values())) for camera, frames in images.items()}
    shares.update({name: float(value) for name, value in other.items()})
    return {
        "maps": {
            camera: {int(position): np.round(grid / peak, 3).tolist() for position, grid in frames.items()}
            for camera, frames in images.items()
        },
        "shares": {name: round(value, 4) for name, value in shares.items()},
        "about": about,
    }
