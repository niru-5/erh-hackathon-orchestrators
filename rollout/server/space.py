"""Run the model server as a private Hugging Face Docker Space on a GPU, only while a session needs it.

    .venv/bin/python -m rollout.server.space deploy   # create/update the Space and start it
    .venv/bin/python -m rollout.server.space start    # start of a session: resume the paused Space
    .venv/bin/python -m rollout.server.space status   # stage, URL and estimated spend vs. the budget
    .venv/bin/python -m rollout.server.space stop     # end of a session: pause it (billing stops)

Spaces bill per minute only while Starting or Running (not while building, sleeping or paused), at
$1.80/h for l40sx1. A paused Space restarts only through `start`, so nothing wakes it by accident;
--sleep-time is a backstop for a session that ends without `stop`. Every start/stop goes into a local
ledger (~/.cache/so101-rollout/space_usage.json), and `deploy`/`start` refuse once the estimated spend
reaches --budget. The estimate counts each session from start to stop, start-up included, so it errs
high; a session left open counts up to now.

Requires `hf auth login` with a write token, a Hugging Face plan that allows GPU Spaces, and that the
account has accepted the license on https://huggingface.co/black-forest-labs/flux-3-action-so101.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token

ROLLOUT = Path(__file__).resolve().parents[1]
LEDGER = Path.home() / ".cache/so101-rollout/space_usage.json"
PRICES = {"l4x1": 0.80, "l40sx1": 1.80, "a100-large": 2.50}  # $/h, huggingface.co/docs/hub/spaces-gpus
# Files the server image needs; tests, robots and the client stay out of the Space.
FILES = [
    "__init__.py",
    "contract.py",
    "codec.py",
    "models/__init__.py",
    "models/flux.py",
    "models/hold.py",
    "server/__init__.py",
    "server/__main__.py",
    "server/app.py",
    "server/requirements.txt",
]
STARTED_STAGES = ("BUILDING", "RUNNING", "RUNNING_BUILDING", "APP_STARTING", "RUNNING_APP_STARTING")
FINAL_STAGES = ("RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "PAUSED", "STOPPED")


class Ledger:
    """Local record of when the Space was started and stopped, to estimate spend."""

    def __init__(self, path: Path = LEDGER):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"sessions": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    def open_session(self) -> dict | None:
        sessions = [s for s in self.data["sessions"] if s.get("stopped") is None]
        return sessions[-1] if sessions else None

    def start(self, space_id: str, hardware: str) -> None:
        if self.open_session() is None:
            self.data["sessions"].append(
                {"space": space_id, "hardware": hardware, "started": time.time(), "stopped": None}
            )
            self.save()

    def stop(self) -> dict | None:
        session = self.open_session()
        if session:
            session["stopped"] = time.time()
            self.save()
        return session

    def hours(self) -> float:
        now = time.time()
        return sum(((s["stopped"] or now) - s["started"]) / 3600 for s in self.data["sessions"])

    def spent(self) -> float:
        now = time.time()
        return sum(
            ((s["stopped"] or now) - s["started"]) / 3600 * PRICES.get(s["hardware"], max(PRICES.values()))
            for s in self.data["sessions"]
        )


def _stage(api: HfApi, space_id: str) -> str:
    stage = api.get_space_runtime(space_id).stage
    return getattr(stage, "value", str(stage))


def _host(api: HfApi, space_id: str) -> str:
    host = getattr(api.space_info(space_id), "host", None)
    return host or f"https://{space_id.replace('/', '-').replace('_', '-').replace('.', '-').lower()}.hf.space"


def _check_budget(ledger: Ledger, budget: float, hardware: str) -> None:
    spent = ledger.spent()
    if spent + PRICES[hardware] * 0.25 > budget:
        raise SystemExit(
            f"estimated spend ${spent:.2f} leaves less than 15 min of {hardware} in the ${budget:.0f} budget; "
            "not starting (raise --budget to override)"
        )


def _wait_running(api: HfApi, args) -> None:
    if args.no_wait:
        return
    print("waiting for the server (first start: image build + ~25 GB weight download, can take 15+ min)...")
    deadline, last = time.monotonic() + args.wait_timeout, None
    while time.monotonic() < deadline:
        stage = _stage(api, args.space_id)
        if stage != last:
            print(f"  {datetime.now():%H:%M:%S} {stage}")
            last = stage
        if stage in FINAL_STAGES:
            break
        time.sleep(15)
    if last != "RUNNING":
        raise SystemExit(
            f"Space is {last}; logs: https://huggingface.co/spaces/{args.space_id}?logs=container "
            "(billing continues while it is starting; run `stop` if you give up)"
        )
    print(f"running: {_host(api, args.space_id)}   (end the session with: python -m rollout.server.space stop)")


def deploy(api: HfApi, args, ledger: Ledger) -> None:
    _check_budget(ledger, args.budget, args.hardware)
    token = get_token()
    if not token:
        raise SystemExit("no HF token found: run `.venv/bin/hf auth login`")
    api.create_repo(args.space_id, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    operations = [CommitOperationAdd(f"rollout/{f}", str(ROLLOUT / f)) for f in FILES]
    operations += [
        CommitOperationAdd("Dockerfile", str(ROLLOUT / "server/Dockerfile")),
        CommitOperationAdd("README.md", str(ROLLOUT / "server/space_README.md")),
    ]
    api.create_commit(args.space_id, operations, repo_type="space", commit_message="Deploy rollout model server")
    api.add_space_secret(args.space_id, "HF_TOKEN", token, description="read access to the gated policy")
    for key, value in (("ROLLOUT_WEIGHTS", args.weights), ("ROLLOUT_REVISION", args.revision)):
        if value:
            api.add_space_variable(args.space_id, key, value)
    api.request_space_hardware(args.space_id, args.hardware, sleep_time=args.sleep_time)
    ledger.start(args.space_id, args.hardware)
    print(f"deployed https://huggingface.co/spaces/{args.space_id} on {args.hardware}")
    _wait_running(api, args)


def start(api: HfApi, args, ledger: Ledger) -> None:
    _check_budget(ledger, args.budget, args.hardware)
    stage = _stage(api, args.space_id)
    if stage in STARTED_STAGES:
        print(f"{args.space_id} is already {stage}")
    else:
        api.restart_space(args.space_id)
    ledger.start(args.space_id, args.hardware)
    _wait_running(api, args)


def stop(api: HfApi, args, ledger: Ledger) -> None:
    api.pause_space(args.space_id)
    session = ledger.stop()
    minutes = f"; this session ran {(session['stopped'] - session['started']) / 60:.0f} min" if session else ""
    print(f"paused {args.space_id} (billing stopped){minutes}")
    status(api, args, ledger)


def status(api: HfApi, args, ledger: Ledger) -> None:
    runtime = api.get_space_runtime(args.space_id)
    print(f"space:   https://huggingface.co/spaces/{args.space_id} ({getattr(runtime.stage, 'value', runtime.stage)})")
    print(f"url:     {_host(api, args.space_id)}")
    print(
        f"spent:   ~${ledger.spent():.2f} of ${args.budget:.0f} ({ledger.hours():.2f} h logged; the estimate errs high)"
    )
    session = ledger.open_session()
    if session:
        print(f"session: open since {datetime.fromtimestamp(session['started']):%Y-%m-%d %H:%M}; run `stop` when done")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live in logs, not only at exit
    parser = argparse.ArgumentParser(
        prog="python -m rollout.server.space", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("deploy", "start", "stop", "status"))
    parser.add_argument("--space-id", help="<user>/<name> (default: <your user>/flux3-so101-rollout)")
    parser.add_argument("--hardware", default="l40sx1", choices=sorted(PRICES), help="GPU flavor (default l40sx1)")
    parser.add_argument("--budget", type=float, default=20.0, help="max estimated total spend in $ (default 20)")
    parser.add_argument(
        "--sleep-time", type=int, default=900, help="backstop: idle seconds before it sleeps (default 900)"
    )
    parser.add_argument("--weights", help="deploy: policy repo (default black-forest-labs/flux-3-action-so101)")
    parser.add_argument("--revision", help="deploy: pin the policy's Hub revision")
    parser.add_argument("--no-wait", action="store_true", help="deploy/start: return without waiting for RUNNING")
    parser.add_argument("--wait-timeout", type=float, default=2400, help="seconds to wait for RUNNING (default 2400)")
    args = parser.parse_args()

    api = HfApi()
    args.space_id = args.space_id or f"{api.whoami()['name']}/flux3-so101-rollout"
    {"deploy": deploy, "start": start, "stop": stop, "status": status}[args.command](api, args, Ledger())


if __name__ == "__main__":
    main()
