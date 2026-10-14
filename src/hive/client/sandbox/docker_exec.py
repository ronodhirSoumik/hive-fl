"""
Docker-based Sandbox Executor.
Executes worker tasks in a network-isolated container (--network none) with read-only
data mounts and strict resource caps.
"""

from __future__ import annotations

import time
from pathlib import Path

from hive.core.errors import SandboxError
from hive.core.schemas import WorkerResult
from hive.client.sandbox.base import SandboxExecutor

try:
    import docker
    from docker.errors import DockerException
    HAS_DOCKER_SDK = True
except ImportError:
    HAS_DOCKER_SDK = False


class DockerExecutor(SandboxExecutor):
    """
    Executes worker tasks inside a Docker container with zero network access (--network none),
    read-only data mounts, and memory/CPU limits.
    """

    def __init__(
        self,
        job_dir: Path | str,
        image_name: str = "hive-worker:latest",
        cpus: float = 1.0,
        memory_mb: int = 2048,
    ) -> None:
        super().__init__(job_dir, isolation="full")
        self.image_name = image_name
        self.cpus = cpus
        self.memory_mb = memory_mb
        self._container = None

    def run(self, timeout_s: float = 600.0) -> WorkerResult:
        if not HAS_DOCKER_SDK:
            raise SandboxError("docker SDK is not installed. Install with: pip install 'hive-fl[docker]'", code="ERR_DOCKER_NOT_INSTALLED")

        try:
            client = docker.from_env()
        except DockerException as e:
            raise SandboxError(f"Cannot connect to Docker daemon: {e}", code="ERR_DOCKER_DAEMON_UNREACHABLE")

        # Map volumes
        job_path_host = str(self.job_dir.resolve())
        volumes = {
            job_path_host: {"bind": "/job", "mode": "rw"},
        }

        # Check data dir from config.json
        cfg_file = self.in_dir / "config.json"
        if cfg_file.exists():
            import json
            cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
            host_data_dir = cfg_data.get("data_dir")
            if host_data_dir and Path(host_data_dir).exists():
                volumes[str(Path(host_data_dir).resolve())] = {"bind": "/data", "mode": "ro"}

        try:
            self._container = client.containers.run(
                image=self.image_name,
                command=["python", "-m", "hive.client.worker.main", "/job"],
                volumes=volumes,
                network_mode="none",  # Strictly no network access
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                pids_limit=256,
                user="1000:1000",
                tmpfs={"/tmp": "rw,size=256m"},
                nano_cpus=int(self.cpus * 1e9),
                mem_limit=f"{self.memory_mb}m",
                detach=True,
                remove=False,
            )

            # Wait with wall-clock timeout
            start_t = time.time()
            while time.time() - start_t < timeout_s:
                self._container.reload()
                if self._container.status in ("exited", "dead"):
                    break
                time.sleep(1.0)

            self._container.reload()
            if self._container.status not in ("exited", "dead"):
                self.kill()
                raise SandboxError(f"Worker container timed out after {timeout_s}s", code="ERR_SANDBOX_TIMEOUT")

            res_file = self.out_dir / "result.json"
            if not res_file.exists():
                logs = self._container.logs().decode("utf-8", errors="replace")
                raise SandboxError(f"Worker container failed without result: {logs[:500]}", code="ERR_SANDBOX_CONTAINER_FAILED")

            result, _, _ = self.parse_result()
            return result

        except Exception as e:
            if isinstance(e, SandboxError):
                raise
            raise SandboxError(f"Docker sandbox execution failed: {e}", code="ERR_SANDBOX_DOCKER_FAILED")

        finally:
            self.kill()

    def suspend(self) -> None:
        if self._container:
            try:
                self._container.pause()
            except Exception:
                pass

    def resume(self) -> None:
        if self._container:
            try:
                self._container.unpause()
            except Exception:
                pass

    def kill(self) -> None:
        if self._container:
            try:
                self._container.kill()
            except Exception:
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None