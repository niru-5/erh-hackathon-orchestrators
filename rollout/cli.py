"""python -m rollout run --robot {so101,dataset} --policy {URL,local:flux,local:hold} ...  (see docs/inference.md)"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from .backends import make_backend
from .history import History
from .loop import JointLimits, RunLog, run_episode

MAIN_DATASET = "niru-5/chess_dataset_c3_6cm_to_d5_10cm_20261017_145741"
logger = logging.getLogger("rollout")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m rollout", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run policy episodes on the arm or on a recorded dataset")
    run.add_argument(
        "--robot",
        choices=("so101", "dataset"),
        required=True,
        help="so101 = the real follower arm; dataset = replay recorded episodes (open loop)",
    )
    run.add_argument(
        "--policy",
        required=True,
        help="server URL (https://<space>.hf.space, http://host:7860) or local:flux / local:hold",
    )
    run.add_argument("--task", help="instruction for the model (default with --robot dataset: the dataset's task)")
    run.add_argument("--episodes", type=int, default=1, help="so101: number of episodes (default 1)")
    run.add_argument("--episode", default="0", help="dataset: comma-separated episode indices (default 0)")
    run.add_argument(
        "--dataset", default=MAIN_DATASET, help=f"dataset repo id for --robot dataset (default {MAIN_DATASET})"
    )
    run.add_argument("--dataset-root", help="local dataset dir (default: lerobot cache, then the Hub)")
    run.add_argument("--episode-time", type=float, default=30.0, help="max seconds per episode (default 30)")
    run.add_argument("--reset-time", type=float, default=15.0, help="so101: seconds between episodes (default 15)")
    safety = run.add_argument_group("safety (so101)")
    safety.add_argument("--dry-run", action="store_true", help="compute everything but never send goals to the arm")
    safety.add_argument(
        "--max-relative-target",
        type=float,
        default=10.0,
        help="max goal distance from the present position per tick, degrees (default 10; 0 = off)",
    )
    safety.add_argument("--limits-from", help="clamp joints to a dataset's recorded range (repo id or local dir)")
    safety.add_argument(
        "--limit-margin", type=float, default=10.0, help="margin around --limits-from range (default 10)"
    )
    transport = run.add_argument_group("transport (remote policy)")
    transport.add_argument("--jpeg-quality", type=int, default=90)
    transport.add_argument(
        "--full-res-images",
        action="store_true",
        help="send camera frames as captured instead of resized to the model's input size",
    )
    transport.add_argument("--timeout", type=float, default=30.0, help="seconds per request (default 30)")
    model = run.add_argument_group("in-process model (local:flux / local:hold)")
    model.add_argument(
        "--weights", help="flux: Hub repo or local package (default black-forest-labs/flux-3-action-so101)"
    )
    model.add_argument("--revision", help="flux: Hub revision to pin")
    model.add_argument("--adapter", help="flux: LeRobot PEFT LoRA to merge, e.g. Gawtam/flux3-chess-lora")
    model.add_argument("--adapter-revision", help="flux: adapter revision to pin")
    model.add_argument("--device", default="cuda")
    model.add_argument("--compile", action="store_true", help="flux: torch.compile the DiT (slow first request)")
    model.add_argument("--hold-delay", type=float, default=0.0, help="hold: simulated inference seconds")
    run.add_argument("--log-dir", type=Path, default=Path("rollout_logs"))
    run.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _backend(args):
    if args.policy.startswith(("http://", "https://")):
        return make_backend(args.policy, jpeg_quality=args.jpeg_quality, timeout_s=args.timeout)
    if args.policy == "local:flux":
        kwargs = {
            "device": args.device,
            "compile_dit": args.compile,
            "revision": args.revision,
            "adapter": args.adapter,
            "adapter_revision": args.adapter_revision,
        }
        if args.weights:
            kwargs["weights"] = args.weights
        return make_backend(args.policy, **kwargs)
    if args.policy == "local:hold":
        return make_backend(args.policy, delay_s=args.hold_delay)
    return make_backend(args.policy)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    for noisy in ("urllib3", "lerobot", "datasets", "fsspec", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    backend = _backend(args)
    spec = backend.spec()
    logger.info(
        "model %s: cameras %s, %d-tick history, %d of %d actions per plan at %.0f Hz",
        spec.name,
        list(spec.camera_keys),
        spec.n_obs_steps,
        spec.n_action_steps,
        spec.chunk_size,
        spec.fps,
    )

    if args.robot == "so101":
        if not args.task:
            raise SystemExit("--task is required with --robot so101")
        from .robots.so101 import SO101Robot

        robot = SO101Robot(
            spec.camera_keys,
            fps=round(spec.fps),
            dry_run=args.dry_run,
            max_relative_target=args.max_relative_target or None,
        )
        episodes = list(range(args.episodes))
    else:
        from .robots.dataset import DatasetRobot

        robot = DatasetRobot(args.dataset, spec.camera_keys, root=args.dataset_root)
        episodes = [int(e) for e in args.episode.split(",")]
        if not args.task:
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

            args.task = str(LeRobotDatasetMetadata(args.dataset, root=args.dataset_root).tasks.index[0])

    limits = JointLimits.from_dataset(args.limits_from, args.limit_margin) if args.limits_from else None
    if limits:
        logger.info("joint limits: %s", limits.describe())
    log = RunLog(args.log_dir / datetime.now().strftime("%Y%m%d_%H%M%S"))
    log.summary({"args": {k: str(v) for k, v in vars(args).items()}, "spec": spec.to_dict()})
    history = History(spec, resize_images=not args.full_res_images)
    max_ticks = round(args.episode_time * spec.fps)

    events, listener = None, None
    if robot.realtime:
        from lerobot.utils.keyboard_input import init_keyboard_listener

        listener, events = init_keyboard_listener()
    results = []
    logger.info("task: %r; logs: %s", args.task, log.directory)
    if args.robot == "so101" and args.dry_run:
        logger.info("DRY RUN: goals are computed and logged but never sent to the arm")
    try:
        robot.connect()
        for n, episode in enumerate(episodes):
            if events and events["stop_recording"]:
                break
            robot.start_episode(episode)
            logger.info(
                "episode %d: running (-> ends it, Esc stops)" if robot.realtime else "episode %d: running", episode
            )
            stats = run_episode(
                robot,
                backend,
                spec,
                history,
                task=args.task,
                episode=episode,
                max_ticks=max_ticks,
                limits=limits,
                log=log,
                events=events,
            )
            result = {"episode": episode, **stats.to_dict(), **robot.summary()}
            results.append(result)
            logger.info("episode %d done: %s", episode, json.dumps(result))
            if events:
                events["exit_early"] = False
            if robot.realtime and n < len(episodes) - 1 and not events["stop_recording"]:
                logger.info("reset the scene: next episode in %.0f s (-> to start now)", args.reset_time)
                deadline = time.monotonic() + args.reset_time
                while time.monotonic() < deadline and not (events["exit_early"] or events["stop_recording"]):
                    time.sleep(0.05)
                events["exit_early"] = False
    finally:
        robot.disconnect()
        backend.close()
        if listener is not None:
            listener.stop()
        log.summary({"args": {k: str(v) for k, v in vars(args).items()}, "spec": spec.to_dict(), "episodes": results})
        log.close()
    print(json.dumps({"logs": str(log.directory), "episodes": results}, indent=2))
    return 0
