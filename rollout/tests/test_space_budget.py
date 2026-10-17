"""Space session ledger and budget guard (no Hub calls)."""

import time

import pytest

from rollout.server.space import Ledger, _check_budget


def test_ledger_counts_sessions_and_open_session_until_now(tmp_path):
    ledger = Ledger(tmp_path / "usage.json")
    hour = 3600
    now = time.time()
    ledger.data["sessions"] = [
        {"space": "u/s", "hardware": "l40sx1", "started": now - 5 * hour, "stopped": now - 3 * hour},  # 2 h
        {"space": "u/s", "hardware": "l40sx1", "started": now - 1 * hour, "stopped": None},  # open: 1 h so far
    ]
    assert ledger.hours() == pytest.approx(3, abs=0.01)
    assert ledger.spent() == pytest.approx(3 * 1.80, abs=0.05)
    ledger.start("u/s", "l40sx1")  # already open: no second session
    assert len(ledger.data["sessions"]) == 2
    session = ledger.stop()
    assert session["stopped"] is not None and ledger.open_session() is None
    assert Ledger(tmp_path / "usage.json").hours() == pytest.approx(3, abs=0.01)  # persisted


def test_budget_guard_refuses_when_less_than_15_min_left(tmp_path):
    ledger = Ledger(tmp_path / "usage.json")
    now = time.time()
    ledger.data["sessions"] = [{"space": "u/s", "hardware": "l40sx1", "started": now - 10 * 3600, "stopped": now}]
    _check_budget(ledger, 20.0, "l40sx1")  # $18.00 spent + $0.45 for 15 min fits
    ledger.data["sessions"][0]["started"] = now - 10.9 * 3600  # $19.62 + $0.45 doesn't
    with pytest.raises(SystemExit, match="budget"):
        _check_budget(ledger, 20.0, "l40sx1")
