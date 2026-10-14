"""
Abstract base class and job-directory layout for client sandbox executors.
Communication between the trusted agent and untrusted worker happens strictly via files.
"""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import torch

from hive.core.errors import SandboxError
from hive.core.schemas import ResourcePolicy, WorkerConfig, WorkerResult
from hive.core.tensors import bytes_to_state_dict, state_dict_to_bytes, validate_update


MAX_UPDATE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB max update safetensors size


class SandboxExecutor(ABC):
    """
    Abstract interface for executing Host-authored code inside an isolated sandbox.
    Communicates with the worker strictly through the job directory file protocol.
    """

    def __init__(self, job_dir: Path | str, isolation: str = "full") -> None:
        self.job_dir = Path(job_dir).resolve()
        self.isolation = isolation
        self.in_dir = self.job_dir / "in"
        self.out_dir = self.job_dir / "out"
        self.pkg_dir = self.in_dir / "pkg"
        self._last_progress_offset = 0

    def prepare(
        self,
        worker_config: WorkerConfig,
        pkg_source_dir: Path | str,
        data_dir: Path | str,
        global_weights: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Creates the standardized job directory structure:
        <job>/in/   config.json, global.safetensors, pkg/ (read-only copy)
        <job>/out/  (initially empty)
        """
        if self.job_dir.exists():
            shutil.rmtree(self.job_dir, ignore_errors=True)

        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write config.json
        cfg_path = self.in_dir / "config.json"
        cfg_dict = worker_config.model_dump()
        cfg_dict["data_dir"] = str(Path(data_dir).resolve())
        cfg_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

        # 2. Copy extracted package contents to <job>/in/pkg/
        src_pkg = Path(pkg_source_dir)
        if src_pkg.exists() and src_pkg.is_dir():
            shutil.copytree(src_pkg, self.pkg_dir, dirs_exist_ok=True)

        # 3. Write global weights if provided (for training tasks)
        if global_weights is not None:
            weights_file = self.in_dir / "global.safetensors"
            weights_file.write_bytes(state_dict_to_bytes(global_weights))

        self._last_progress_offset = 0

    def setup_job_directory(
        self,
        worker_config: WorkerConfig,
        pkg_source_dir: Path | str,
        data_dir: Path | str,
        global_weights: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Backward-compatible alias for prepare()."""
        self.prepare(worker_config, pkg_source_dir, data_dir, global_weights)

    def progress_stream(self) -> list[dict[str, Any]]:
        """
        Reads any newly appended progress records from <job>/out/progress.jsonl.
        Returns a list of parsed JSON records since the previous call.
        """
        prog_file = self.out_dir / "progress.jsonl"
        if not prog_file.exists():
            return []

        records: list[dict[str, Any]] = []
        try:
            with open(prog_file, "r", encoding="utf-8") as f:
                f.seek(self._last_progress_offset)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError:
                            continue
                self._last_progress_offset = f.tell()
        except Exception:
            pass
        return records

    def parse_result(
        self,
        template: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[WorkerResult, dict[str, torch.Tensor] | None, bytes | None]:
        """
        Parses worker outputs from <job>/out/result.json and <job>/out/update.safetensors.
        Strictly validates updates against global template if present, rejects NaNs/Infs,
        and enforces size limits.
        """
        res_file = self.out_dir / "result.json"
        if not res_file.exists():
            raise SandboxError("Worker failed to produce result.json", code="ERR_WORKER_NO_RESULT")

        try:
            res_dict = json.loads(res_file.read_text(encoding="utf-8"))
            # Only known WorkerResult fields allowed; free-text strings not forwarded
            result = WorkerResult.model_validate(res_dict)
        except Exception as e:
            raise SandboxError(f"Malformed worker result.json: {e}", code="ERR_WORKER_RESULT_MALFORMED")

        if result.status != "ok":
            return result, None, None

        # Check for update.safetensors
        delta_file = self.out_dir / "update.safetensors"
        delta_dict = None
        delta_bytes = None
        if delta_file.exists():
            if delta_file.stat().st_size > MAX_UPDATE_SIZE_BYTES:
                raise SandboxError(
                    f"Worker update exceeds maximum allowed size ({delta_file.stat().st_size} > {MAX_UPDATE_SIZE_BYTES} bytes)",
                    code="ERR_UPDATE_OVERSIZED",
                )
            delta_bytes = delta_file.read_bytes()
            try:
                delta_dict = bytes_to_state_dict(delta_bytes)
            except Exception as e:
                raise SandboxError(f"Malformed safetensors in update: {e}", code="ERR_UPDATE_MALFORMED")

            # Check for NaNs/Infs and validate against global template
            if template is not None:
                try:
                    validate_update(delta_dict, template)
                except Exception as e:
                    raise SandboxError(f"Update failed validation against template: {e}", code="ERR_UPDATE_VALIDATION_FAILED")
            else:
                for name, t in delta_dict.items():
                    if not torch.all(torch.isfinite(t)):
                        raise SandboxError(f"Tensor '{name}' contains NaN or Inf values", code="ERR_UPDATE_NAN_INF")

        return result, delta_dict, delta_bytes

    @abstractmethod
    def run(self, timeout_s: float = 600.0) -> WorkerResult:
        """Executes the worker process/container to completion."""
        pass

    @abstractmethod
    def suspend(self) -> None:
        """Freezes/pauses the active worker execution."""
        pass

    @abstractmethod
    def resume(self) -> None:
        """Unfreezes/resumes the paused worker execution."""
        pass

    @abstractmethod
    def kill(self) -> None:
        """Immediately terminates the worker."""
        pass

    def cleanup(self) -> None:
        """Removes the job directory and volatile files."""
        if self.job_dir.exists():
            shutil.rmtree(self.job_dir, ignore_errors=True)