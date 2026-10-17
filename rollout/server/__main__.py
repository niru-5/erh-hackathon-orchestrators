"""python -m rollout.server --model {flux,hold} [--port 7860]: load the model, warm it up, then serve it."""

import argparse
import logging
import os
import time

import numpy as np

from ..contract import Window
from ..models import MODELS, make_model
from .app import serve

logger = logging.getLogger("rollout.server")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rollout.server", description=__doc__)
    parser.add_argument("--model", choices=MODELS, default=os.environ.get("ROLLOUT_MODEL", "flux"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument(
        "--weights",
        default=os.environ.get("ROLLOUT_WEIGHTS") or None,
        help="flux: Hub repo or local package (env ROLLOUT_WEIGHTS)",
    )
    parser.add_argument("--revision", default=os.environ.get("ROLLOUT_REVISION") or None, help="flux: Hub revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile",
        action="store_true",
        default=os.environ.get("ROLLOUT_COMPILE") == "1",
        help="flux: torch.compile the DiT (env ROLLOUT_COMPILE=1)",
    )
    parser.add_argument("--offload-text-encoder", action="store_true", help="flux: keep the text encoder on CPU")
    parser.add_argument("--hold-delay", type=float, default=0.0, help="hold: simulated inference seconds")
    parser.add_argument("--no-warmup", action="store_true", help="skip the warm-up prediction before listening")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    started = time.perf_counter()
    if args.model == "flux":
        kwargs = {
            "revision": args.revision,
            "device": args.device,
            "compile_dit": args.compile,
            "offload_text_encoder": args.offload_text_encoder,
        }
        if args.weights:
            kwargs["weights"] = args.weights
        model = make_model("flux", **kwargs)
    else:
        model = make_model("hold", delay_s=args.hold_delay)
    spec = model.spec()
    logger.info("loaded %s in %.0f s", spec.name, time.perf_counter() - started)
    if not args.no_warmup:
        h, w = spec.image_hw
        n, d = spec.n_obs_steps, len(spec.action_names)
        blank = np.zeros((h, w, 3), np.uint8)
        window = Window(
            "warm-up",
            np.zeros((n, d), np.float32),
            np.zeros((n, d), np.float32),
            {c: {i: blank for i in spec.snapshot_indices} for c in spec.camera_keys},
        )
        for i in range(2):
            plan = model.predict(window)
            logger.info("warm-up %d: %.0f ms", i + 1, plan.timing.get("inference_ms", float("nan")))
    serve(model, args.host, args.port)


if __name__ == "__main__":
    main()
