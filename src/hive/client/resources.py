"""
Resource controller and idle governor for client training execution.
Enforces user preferences: idle detection, CPU caps, battery pauses, schedule windows,
and maximum contributed rounds.
"""

from __future__ import annotations

import asyncio
import datetime
import threading
import time
from typing import Any, Callable

from hive.client.config_store import ClientConfigStore
from hive.client.idle import is_on_battery, system_cpu_excluding, user_idle_seconds
from hive.client.sandbox.base import SandboxExecutor
from hive.core.logging import get_logger
from hive.core.schemas import DPPolicy, ResourcePolicy, WorkerConfig

logger = get_logger("hive.client.resources")


def is_within_schedule(schedule: dict[str, Any], current_dt: datetime.datetime | None = None) -> bool:
    """
    Evaluates whether current time falls within scheduled start/end HH:MM window.
    Handles overnight spans (e.g. 22:00 to 07:00).
    """
    if not schedule.get("enabled", False):
        return True

    start_str = schedule.get("start", "22:00")
    end_str = schedule.get("end", "07:00")

    try:
        dt = current_dt or datetime.datetime.now()
        cur_t = dt.time()
        start_t = datetime.datetime.strptime(start_str, "%H:%M").time()
        end_t = datetime.datetime.strptime(end_str, "%H:%M").time()

        if start_t <= end_t:
            return start_t <= cur_t <= end_t
        else:
            # Overnight span (e.g. 22:00 to 06:00)
            return cur_t >= start_t or cur_t <= end_t
    except Exception as e:
        logger.warning("Failed to parse schedule window", error=str(e))
        return True


class ResourceController:
    """
    Govern when client training can proceed and adaptively suspends/resumes execution.
    """

    def __init__(
        self,
        policy: ResourcePolicy | None = None,
        config_store: ClientConfigStore | None = None,
        clock_provider: Callable[[], float] = time.time,
        battery_provider: Callable[[], bool] = is_on_battery,
        cpu_provider: Callable[[], float] = system_cpu_excluding,
        idle_provider: Callable[[], float | None] = user_idle_seconds,
    ) -> None:
        self.config_store = config_store or ClientConfigStore()
        self.policy = policy or self.config_store.load_resource_policy()

        # Injected sensors for testability
        self.clock_provider = clock_provider
        self.battery_provider = battery_provider
        self.cpu_provider = cpu_provider
        self.idle_provider = idle_provider

        # State tracking
        self.is_user_paused: bool = False
        self.rounds_contributed: int = 0
        self.active_executor: SandboxExecutor | None = None
        self.is_suspended: bool = False

        # Monitor thread
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    def update_policy(self, new_policy: ResourcePolicy) -> None:
        """Updates active policy and persists it to disk."""
        self.policy = new_policy
        self.config_store.save_resource_policy(new_policy)
        logger.info(f"Updated resource policy: max_cpu={new_policy.max_cpu_percent}, idle_only={new_policy.only_when_idle}")

    def pause(self) -> None:
        """Manually pauses contributor training."""
        self.is_user_paused = True
        if self.active_executor and not self.is_suspended:
            self.active_executor.suspend()
            self.is_suspended = True
        logger.info("Training paused by user")

    def resume(self) -> None:
        """Resumes contributor training if resource criteria are met."""
        self.is_user_paused = False
        can, _ = self.can_train()
        if self.active_executor and self.is_suspended and can:
            self.active_executor.resume()
            self.is_suspended = False
        logger.info("Training resumed by user")

    def record_round_completed(self) -> None:
        """Increments contributed round counter."""
        self.rounds_contributed += 1
        logger.info(f"Recorded contributed round: total={self.rounds_contributed}")

    def can_train(self) -> tuple[bool, str]:
        """
        Evaluates all resource gating rules.
        Returns (True, reason) if training is permitted, or (False, reason) if blocked.
        """
        if self.is_user_paused:
            return False, "Paused by user"

        if self.policy.max_rounds > 0 and self.rounds_contributed >= self.policy.max_rounds:
            return False, f"Contributed max configured rounds ({self.rounds_contributed}/{self.policy.max_rounds})"

        # Schedule check
        if self.policy.schedule and self.policy.schedule.get("enabled", False):
            now_dt = datetime.datetime.fromtimestamp(self.clock_provider())
            if not is_within_schedule(self.policy.schedule, now_dt):
                return False, "Outside scheduled training hours"

        # Battery check
        if self.policy.pause_on_battery and self.battery_provider():
            return False, "Paused on battery power"

        # Idle checks
        if self.policy.only_when_idle:
            user_idle = self.idle_provider()
            if user_idle is not None:
                req_idle_s = self.policy.idle_minutes * 60.0
                if user_idle < req_idle_s:
                    return False, f"User active (idle only {int(user_idle)}s < {int(req_idle_s)}s)"

            cpu = self.cpu_provider()
            if cpu > self.policy.idle_cpu_threshold:
                return False, f"System busy (CPU {cpu:.1f}% > {self.policy.idle_cpu_threshold:.1f}%)"

        return True, "All resource conditions met"

    def set_active_executor(self, executor: SandboxExecutor | None) -> None:
        """Registers the currently executing sandbox executor for suspension supervision."""
        self.active_executor = executor
        self.is_suspended = False

    def check_and_apply_governance(self) -> None:
        """Synchronously checks can_train and suspends or resumes active executor accordingly."""
        can, reason = self.can_train()
        if not can:
            if self.active_executor and not self.is_suspended:
                try:
                    self.active_executor.suspend()
                    self.is_suspended = True
                    logger.info(f"Suspended worker execution: reason={reason}")
                except Exception as e:
                    logger.warning(f"Failed to suspend executor: {e}")
        else:
            if self.active_executor and self.is_suspended:
                try:
                    self.active_executor.resume()
                    self.is_suspended = False
                    logger.info(f"Resumed worker execution: reason={reason}")
                except Exception as e:
                    logger.warning(f"Failed to resume executor: {e}")

    def start_monitor(self, interval_s: float = 2.0) -> None:
        """Starts background monitor thread checking resource condition flips."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()

        def _monitor_loop() -> None:
            while not self._stop_event.is_set():
                self.check_and_apply_governance()
                self._stop_event.wait(interval_s)

        self._monitor_thread = threading.Thread(target=_monitor_loop, daemon=True, name="ResourceMonitor")
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        """Stops background monitor thread."""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)
            self._monitor_thread = None

    def build_worker_config(
        self,
        task: str,
        hyperparameters: dict[str, Any] | None = None,
        dp: DPPolicy | None = None,
        seed: int = 42,
    ) -> WorkerConfig:
        """Translates current ResourcePolicy into WorkerConfig parameters."""
        return WorkerConfig(
            task=task,  # type: ignore[arg-type]
            hyperparameters=hyperparameters or {},
            dp=dp,
            max_cpu_percent=self.policy.max_cpu_percent,
            seed=seed,
            use_gpu=self.policy.use_gpu,
        )
