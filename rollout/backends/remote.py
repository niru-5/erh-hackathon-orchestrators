"""Call a model behind rollout.server over HTTP: a Hugging Face Space, an Inference Endpoint or any host."""

import json
import logging
import os
import time
from urllib.parse import urlparse

import requests

from ..codec import DEFAULT_JPEG_QUALITY, decode_plan, encode_window
from ..contract import Plan, PolicySpec, Window

logger = logging.getLogger(__name__)

HF_HOSTS = (".hf.space", ".huggingface.cloud", "huggingface.co")
WAKING_STATUS = (502, 503, 504)  # a sleeping Space / scaled-to-zero endpoint answers these while it starts


def _hf_token() -> str | None:
    from huggingface_hub import get_token

    return os.environ.get("HF_TOKEN") or get_token()


class RemoteBackend:
    def __init__(
        self,
        url: str,
        *,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        timeout_s: float = 30.0,
        wake_timeout_s: float = 900.0,
    ):
        self.url = url.rstrip("/")
        self.jpeg_quality = jpeg_quality
        self.timeout_s = timeout_s
        self.wake_timeout_s = wake_timeout_s
        self.session = requests.Session()
        host = urlparse(self.url).hostname or ""
        if any(host.endswith(h) for h in HF_HOSTS):
            token = _hf_token()
            if token:
                self.session.headers["Authorization"] = f"Bearer {token}"
            else:
                logger.warning("no Hugging Face token found (hf auth login); private Spaces will refuse requests")
        if os.environ.get("ROLLOUT_API_KEY"):
            self.session.headers["X-Rollout-Key"] = os.environ["ROLLOUT_API_KEY"]
        self._spec: PolicySpec | None = None

    def spec(self) -> PolicySpec:
        """Fetch the model contract, waiting for a sleeping Space to start (that can take minutes)."""
        if self._spec is None:
            deadline = time.monotonic() + self.wake_timeout_s
            while True:
                try:
                    response = self.session.get(f"{self.url}/spec", timeout=self.timeout_s)
                    if response.status_code not in WAKING_STATUS:
                        break
                    reason = f"HTTP {response.status_code}"
                except requests.ConnectionError as error:
                    reason = type(error).__name__
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self.url} did not become ready within {self.wake_timeout_s:.0f} s")
                logger.info("waiting for %s to start (%s)...", self.url, reason)
                time.sleep(10)
            _raise_for_status(response)
            self._spec = PolicySpec.from_dict(response.json())
        return self._spec

    def predict(self, window: Window) -> Plan:
        body = json.dumps(encode_window(window, self.jpeg_quality)).encode()
        started = time.perf_counter()
        response = self.session.post(
            f"{self.url}/predict", data=body, headers={"Content-Type": "application/json"}, timeout=self.timeout_s
        )
        _raise_for_status(response)
        plan = decode_plan(response.json())
        plan.timing["client_total_ms"] = (time.perf_counter() - started) * 1e3
        plan.timing["request_bytes"] = len(body)
        return plan

    def close(self) -> None:
        self.session.close()


def _raise_for_status(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise RuntimeError(f"{response.request.method} {response.url} -> HTTP {response.status_code}: {str(detail)[:500]}")
