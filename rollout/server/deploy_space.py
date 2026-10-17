"""Deploy rollout.server with FLUX 3 Action to a private Hugging Face Docker Space on a GPU.

    .venv/bin/python -m rollout.server.deploy_space --space-id <user>/flux3-so101-rollout

This costs money while the Space runs (l40sx1 is billed per hour). The Space sleeps after --sleep-time
seconds without requests; the next request wakes it (a few minutes: image start + weight download).
Pause it by hand with `--pause`. Requires `hf auth login` with a write token, and that the account has
accepted the license on https://huggingface.co/black-forest-labs/flux-3-action-so101.
"""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token

ROLLOUT = Path(__file__).resolve().parents[1]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rollout.server.deploy_space",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--space-id", help="<user>/<name> (default: <your user>/flux3-so101-rollout)")
    parser.add_argument("--hardware", default="l40sx1", help="GPU flavor, needs ~32 GB (default l40sx1, 48 GB)")
    parser.add_argument(
        "--sleep-time", type=int, default=3600, help="seconds idle before the Space sleeps (default 3600)"
    )
    parser.add_argument(
        "--weights", help="policy repo or revision-pinned weights (default black-forest-labs/flux-3-action-so101)"
    )
    parser.add_argument("--revision", help="pin the policy's Hub revision")
    parser.add_argument("--public", action="store_true", help="make the Space public (default private)")
    parser.add_argument(
        "--no-token-secret",
        action="store_true",
        help="don't store your local HF token as the Space's HF_TOKEN secret (set it in the Space settings)",
    )
    parser.add_argument("--pause", action="store_true", help="only pause the Space (stops billing) and exit")
    args = parser.parse_args()

    api = HfApi()
    space_id = args.space_id or f"{api.whoami()['name']}/flux3-so101-rollout"
    if args.pause:
        api.pause_space(space_id)
        print(f"paused {space_id}; restart it from the Space settings or with HfApi().restart_space()")
        return

    secrets = []
    if not args.no_token_secret:
        token = get_token()
        if not token:
            raise SystemExit("no HF token found: run `.venv/bin/hf auth login` or pass --no-token-secret")
        secrets.append({"key": "HF_TOKEN", "value": token})
    variables = [
        {"key": k, "value": v} for k, v in (("ROLLOUT_WEIGHTS", args.weights), ("ROLLOUT_REVISION", args.revision)) if v
    ]

    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=not args.public, exist_ok=True)
    operations = [CommitOperationAdd(f"rollout/{f}", str(ROLLOUT / f)) for f in FILES]
    operations += [
        CommitOperationAdd("Dockerfile", str(ROLLOUT / "server/Dockerfile")),
        CommitOperationAdd("README.md", str(ROLLOUT / "server/space_README.md")),
    ]
    api.create_commit(space_id, operations, repo_type="space", commit_message="Deploy rollout model server")
    for secret in secrets:
        api.add_space_secret(space_id, secret["key"], secret["value"], description="read access to the gated policy")
    for variable in variables:
        api.add_space_variable(space_id, variable["key"], variable["value"])
    runtime = api.request_space_hardware(space_id, args.hardware, sleep_time=args.sleep_time)
    host = api.space_info(space_id).host or f"https://{space_id.replace('/', '-').replace('_', '-').lower()}.hf.space"
    print(f"Space: https://huggingface.co/spaces/{space_id} (hardware {args.hardware}, stage {runtime.stage})")
    print(
        f"Building; the first start downloads the weights. Then:\n"
        f"  .venv/bin/python -m rollout run --robot dataset --policy {host} --episode 0"
    )


if __name__ == "__main__":
    main()
