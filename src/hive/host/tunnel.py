"""
Reachability, NAT traversal, and Tunnel Manager for the Hive Host.
Supports Local/LAN detection, Cloudflare Quick Tunnels, manual public URLs,
and canonical join URI construction.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import socket
import subprocess
from typing import Any

from hive.core.errors import ConfigurationError
from hive.core.logging import get_logger
from hive.core.uri import build_hive_uri

logger = get_logger("hive.host.tunnel")

CLOUDFLARE_REGEX = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def detect_lan_ip() -> str:
    """Detects local IPv4 address using standard socket connection to an external target."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually transmit packets
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def check_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a TCP port is currently free for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


class TunnelManager:
    """
    Manages host exposure via direct LAN, manual public URL, or Cloudflare Quick Tunnel.
    """

    def __init__(
        self,
        mode: str = "none",
        api_port: int = 9000,
        manual_public_url: str | None = None,
    ) -> None:
        self.mode = mode.lower()
        self.api_port = api_port
        self.manual_public_url = manual_public_url
        self.public_url: str = ""
        self.is_tls: bool = False
        self._process: subprocess.Popen[str] | None = None

    async def start(self) -> str:
        """
        Starts the tunnel and returns the public HTTP(S) URL.
        """
        if self.mode == "manual":
            if not self.manual_public_url:
                raise ConfigurationError("Mode 'manual' requires --public-url to be specified")
            self.public_url = self.manual_public_url.rstrip("/")
            self.is_tls = self.public_url.startswith("https://")
            logger.info(f"Using manual public URL: {self.public_url} (TLS: {self.is_tls})")
            return self.public_url

        elif self.mode == "cloudflared":
            # Check binary
            if not shutil.which("cloudflared"):
                raise ConfigurationError(
                    "cloudflared binary not found in PATH. Install cloudflared or run with --tunnel none",
                    code="ERR_CLOUDFLARED_NOT_FOUND",
                )
            cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{self.api_port}"]
            logger.info(f"Launching Cloudflare quick tunnel: {' '.join(cmd)}")
            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as e:
                raise ConfigurationError(f"Failed to spawn cloudflared: {e}")

            # Read stream until URL is found
            loop = asyncio.get_running_loop()
            url_future: asyncio.Future[str] = loop.create_future()

            def _read_output() -> None:
                if not self._process or not self._process.stdout:
                    return
                for line in self._process.stdout:
                    match = CLOUDFLARE_REGEX.search(line)
                    if match and not url_future.done():
                        loop.call_soon_threadsafe(url_future.set_result, match.group(0))
                        break

            asyncio.get_event_loop().run_in_executor(None, _read_output)

            try:
                self.public_url = await asyncio.wait_for(url_future, timeout=25.0)
                self.is_tls = True
                logger.info(f"Cloudflare tunnel established: {self.public_url}")
                return self.public_url
            except asyncio.TimeoutError:
                self.stop()
                raise ConfigurationError("Timed out waiting for cloudflared quick tunnel URL (25s)")

        else:
            # Mode "none" (Direct LAN / localhost)
            if self.manual_public_url:
                self.public_url = self.manual_public_url.rstrip("/")
                self.is_tls = self.public_url.startswith("https://")
            else:
                lan_ip = detect_lan_ip()
                self.public_url = f"http://{lan_ip}:{self.api_port}"
                self.is_tls = False
            logger.info(f"Using LAN endpoint: {self.public_url} (TLS: {self.is_tls})")
            return self.public_url

    def build_join_uri(self, session_id: str, join_token: str, fingerprint: str) -> str:
        """Constructs canonical hive:// join URI from current tunnel and session state."""
        auth = self.public_url.replace("https://", "").replace("http://", "")
        return build_hive_uri(
            authority=auth,
            session_id=session_id,
            join_token=join_token,
            fingerprint=fingerprint,
            tls=self.is_tls,
        )

    def stop(self) -> None:
        """Terminates tunnel background subprocess if running."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None


def run_doctor() -> dict[str, Any]:
    """Runs system diagnostics on environment, Docker, Python version, and network."""
    import sys
    import torch

    docker_present = shutil.which("docker") is not None
    cloudflared_present = shutil.which("cloudflared") is not None
    api_free = check_port_free(9000)
    admin_free = check_port_free(8000)
    lan_ip = detect_lan_ip()

    py_version = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 10)

    return {
        "python": {"version": py_version, "ok": py_ok},
        "torch": {"version": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "docker": {"installed": docker_present},
        "cloudflared": {"installed": cloudflared_present},
        "ports": {"api_port_9000_free": api_free, "admin_port_8000_free": admin_free},
        "network": {"lan_ip": lan_ip},
    }