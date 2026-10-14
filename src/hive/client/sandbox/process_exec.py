"""
Process-based Sandbox Executor.
Executes worker tasks in a dedicated subprocess with scrubbed environment variables
and process suspension controls.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from hive.core.errors import SandboxError
from hive.core.schemas import WorkerResult
from hive.client.sandbox.base import SandboxExecutor

# Safe environment variables allowed to pass through to worker subprocess
SAFE_ENV_KEYS = {
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "PYTHONPATH",
    "PYTHONHOME",
    "COMSPEC",
    "PATHEXT",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
}


class ProcessExecutor(SandboxExecutor):
    """
    Executes worker tasks as a local child process.
    Tagged with isolation="reduced" since it shares the local filesystem/network namespaces.
    """

    def __init__(self, job_dir: Path | str, max_ram_mb: int | None = None) -> None:
        super().__init__(job_dir, isolation="reduced")
        self.max_ram_mb = max_ram_mb
        self._process: subprocess.Popen[str] | None = None

    def _get_scrubbed_env(self) -> dict[str, str]:
        """Constructs an environment dictionary with sensitive tokens scrubbed."""
        clean: dict[str, str] = {}
        for k, v in os.environ.items():
            if k.upper() in SAFE_ENV_KEYS:
                clean[k] = v
        # Ensure current python path includes package src
        cur_pythonpath = clean.get("PYTHONPATH", "")
        cwd = str(Path.cwd().resolve())
        src_path = str((Path.cwd() / "src").resolve())
        clean["PYTHONPATH"] = f"{src_path}{os.pathsep}{cur_pythonpath}" if cur_pythonpath else src_path
        clean["PYTHONUNBUFFERED"] = "1"
        return clean

    def run(self, timeout_s: float = 600.0) -> WorkerResult:
        cmd = [sys.executable, "-m", "hive.client.worker.main", str(self.job_dir)]
        env = self._get_scrubbed_env()

        preexec = None
        if sys.platform != "win32" and self.max_ram_mb:
            def _set_rlimits() -> None:
                try:
                    import resource
                    # PyTorch libtorch shared library requires virtual address space for mmap (~8GB)
                    min_as_bytes = 8 * 1024 * 1024 * 1024
                    ram_bytes = max(self.max_ram_mb * 1024 * 1024, min_as_bytes)
                    resource.setrlimit(resource.RLIMIT_AS, (ram_bytes, ram_bytes))
                except Exception:
                    pass
            preexec = _set_rlimits

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self.job_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=preexec,
            )

            stdout, stderr = self._process.communicate(timeout=timeout_s)

            if self._process.returncode != 0:
                # Log stderr locally, do not forward raw trace to network
                err_snippet = stderr.strip() if stderr else f"Process exited with code {self._process.returncode}"
                # If result.json was written, parse it; else raise SandboxError
                res_file = self.out_dir / "result.json"
                if res_file.exists():
                    result, _, _ = self.parse_result()
                    return result
                raise SandboxError(f"Worker process failed: {err_snippet}", code="ERR_SANDBOX_PROCESS_FAILED")

            result, _, _ = self.parse_result()
            return result

        except subprocess.TimeoutExpired:
            self.kill()
            raise SandboxError(f"Worker process timed out after {timeout_s}s", code="ERR_SANDBOX_TIMEOUT")
        finally:
            self._process = None

    def suspend(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                p = psutil.Process(self._process.pid)
                p.suspend()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def resume(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                p = psutil.Process(self._process.pid)
                p.resume()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def kill(self) -> None:
        if self._process:
            try:
                p = psutil.Process(self._process.pid)
                for child in p.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self._process = None