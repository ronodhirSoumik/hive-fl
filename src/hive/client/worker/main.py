"""
In-Sandbox Worker main executable entrypoint.
Invoked via: python -m hive.client.worker.main <job_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from hive.client.worker.trainer import run_worker_task


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python -m hive.client.worker.main <job_directory>\n")
        sys.exit(1)

    job_dir = Path(sys.argv[1])
    run_worker_task(job_dir)


if __name__ == "__main__":
    main()