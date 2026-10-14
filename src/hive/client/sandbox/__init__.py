"""
Sandbox executor module for isolated client worker execution.
"""

from pathlib import Path

from hive.client.sandbox.base import SandboxExecutor
from hive.client.sandbox.docker_exec import HAS_DOCKER_SDK, DockerExecutor
from hive.client.sandbox.process_exec import ProcessExecutor


def get_sandbox_executor(
    mode: str = "auto",
    job_dir: Path | str = ".hive/client/job",
    cpus: float = 1.0,
    memory_mb: int = 2048,
) -> SandboxExecutor:
    """
    Selects and instantiates the appropriate sandbox executor based on policy and environment.
    """
    selected = mode.lower().strip()
    if selected == "docker":
        return DockerExecutor(job_dir=job_dir, cpus=cpus, memory_mb=memory_mb)
    elif selected == "process":
        return ProcessExecutor(job_dir=job_dir, max_ram_mb=memory_mb)
    elif selected == "auto":
        # Check if Docker is available and daemon is responding
        if HAS_DOCKER_SDK:
            try:
                import docker
                client = docker.from_env()
                client.ping()
                return DockerExecutor(job_dir=job_dir, cpus=cpus, memory_mb=memory_mb)
            except Exception:
                pass
        return ProcessExecutor(job_dir=job_dir, max_ram_mb=memory_mb)
    else:
        raise ValueError(f"Unknown sandbox mode: '{mode}'. Must be 'auto', 'docker', or 'process'.")


__all__ = [
    "SandboxExecutor",
    "ProcessExecutor",
    "DockerExecutor",
    "get_sandbox_executor",
]