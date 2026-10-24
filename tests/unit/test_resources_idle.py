"""
Unit tests for ResourceController and Idle Detection (T12).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from hive.client.config_store import ClientConfigStore
from hive.client.resources import ResourceController, is_within_schedule
from hive.client.sandbox.base import SandboxExecutor
from hive.core.schemas import ResourcePolicy, WorkerResult


class MockExecutor(SandboxExecutor):
    def __init__(self) -> None:
        super().__init__("/tmp/mock_job", isolation="mock")
        self.suspend_count = 0
        self.resume_count = 0

    def run(self, timeout_s: float = 600.0) -> WorkerResult:
        return WorkerResult(status="ok")

    def suspend(self) -> None:
        self.suspend_count += 1

    def resume(self) -> None:
        self.resume_count += 1

    def kill(self) -> None:
        pass


def test_schedule_window_evaluation() -> None:
    """Tests schedule window parsing including overnight spans."""
    # Disabled schedule is always within window
    assert is_within_schedule({"enabled": False}) is True

    # Standard daytime span (09:00 - 17:00)
    day_sched = {"enabled": True, "start": "09:00", "end": "17:00"}
    assert is_within_schedule(day_sched, datetime.datetime(2026, 1, 1, 12, 0)) is True
    assert is_within_schedule(day_sched, datetime.datetime(2026, 1, 1, 8, 30)) is False
    assert is_within_schedule(day_sched, datetime.datetime(2026, 1, 1, 18, 0)) is False

    # Overnight span (22:00 - 06:00)
    night_sched = {"enabled": True, "start": "22:00", "end": "06:00"}
    assert is_within_schedule(night_sched, datetime.datetime(2026, 1, 1, 23, 30)) is True
    assert is_within_schedule(night_sched, datetime.datetime(2026, 1, 1, 4, 15)) is True
    assert is_within_schedule(night_sched, datetime.datetime(2026, 1, 1, 14, 0)) is False


def test_can_train_gating_conditions(tmp_path: Path) -> None:
    """Verifies all gating criteria of can_train()."""
    store = ClientConfigStore(tmp_path)

    # 1. User pause
    ctrl = ResourceController(config_store=store)
    ctrl.pause()
    can, reason = ctrl.can_train()
    assert can is False
    assert "Paused by user" in reason
    ctrl.resume()

    # 2. Max rounds limit
    pol_max = ResourcePolicy(max_rounds=2)
    ctrl_rounds = ResourceController(policy=pol_max, config_store=store)
    ctrl_rounds.rounds_contributed = 2
    can, reason = ctrl_rounds.can_train()
    assert can is False
    assert "Contributed max configured rounds" in reason

    # 3. Battery pause
    ctrl_batt = ResourceController(
        policy=ResourcePolicy(pause_on_battery=True),
        battery_provider=lambda: True,  # on battery
        config_store=store,
    )
    can, reason = ctrl_batt.can_train()
    assert can is False
    assert "Paused on battery" in reason

    # 4. User active (idle minutes)
    ctrl_active = ResourceController(
        policy=ResourcePolicy(only_when_idle=True, idle_minutes=2.0),
        battery_provider=lambda: False,
        idle_provider=lambda: 30.0,  # 30s idle < 120s required
        config_store=store,
    )
    can, reason = ctrl_active.can_train()
    assert can is False
    assert "User active" in reason

    # 5. CPU busy
    ctrl_busy = ResourceController(
        policy=ResourcePolicy(only_when_idle=True, idle_cpu_threshold=25.0),
        battery_provider=lambda: False,
        idle_provider=lambda: 300.0,  # idle enough
        cpu_provider=lambda: 60.0,  # 60% > 25% threshold
        config_store=store,
    )
    can, reason = ctrl_busy.can_train()
    assert can is False
    assert "System busy" in reason

    # 6. All conditions met
    ctrl_good = ResourceController(
        policy=ResourcePolicy(only_when_idle=True, idle_cpu_threshold=25.0, pause_on_battery=True),
        battery_provider=lambda: False,
        idle_provider=lambda: 300.0,
        cpu_provider=lambda: 10.0,
        config_store=store,
    )
    can, reason = ctrl_good.can_train()
    assert can is True
    assert "All resource conditions met" in reason


def test_mid_round_executor_suspension(tmp_path: Path) -> None:
    """Verifies that check_and_apply_governance suspends and resumes active executor."""
    store = ClientConfigStore(tmp_path)
    mock_exec = MockExecutor()

    condition_ok = [True]

    ctrl = ResourceController(
        battery_provider=lambda: not condition_ok[0],
        policy=ResourcePolicy(pause_on_battery=True),
        config_store=store,
    )
    ctrl.set_active_executor(mock_exec)

    # Condition is good -> no suspension
    ctrl.check_and_apply_governance()
    assert mock_exec.suspend_count == 0
    assert ctrl.is_suspended is False

    # Condition flips to busy/battery -> triggers suspend
    condition_ok[0] = False
    ctrl.check_and_apply_governance()
    assert mock_exec.suspend_count == 1
    assert ctrl.is_suspended is True

    # Second check while still bad -> does not duplicate suspend
    ctrl.check_and_apply_governance()
    assert mock_exec.suspend_count == 1

    # Condition recovers -> triggers resume
    condition_ok[0] = True
    ctrl.check_and_apply_governance()
    assert mock_exec.resume_count == 1
    assert ctrl.is_suspended is False


def test_policy_persistence(tmp_path: Path) -> None:
    """Verifies updating and loading policy from config store."""
    store = ClientConfigStore(tmp_path)
    ctrl = ResourceController(config_store=store)

    new_pol = ResourcePolicy(max_cpu_percent=75, max_rounds=5, use_gpu=True)
    ctrl.update_policy(new_pol)

    loaded = store.load_resource_policy()
    assert loaded.max_cpu_percent == 75
    assert loaded.max_rounds == 5
    assert loaded.use_gpu is True
