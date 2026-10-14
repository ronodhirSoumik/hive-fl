"""
System and user activity sensors for idle detection.
Detects user inactivity across Windows, macOS, and Linux platforms.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Callable

import psutil


def get_windows_idle_seconds() -> float | None:
    """Uses Win32 GetLastInputInfo to determine user keyboard/mouse idle time in seconds."""
    try:
        import ctypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint),
                ("dwTime", ctypes.c_uint),
            ]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
            return max(0.0, float(millis) / 1000.0)
    except Exception:
        pass
    return None


def get_macos_idle_seconds() -> float | None:
    """Uses ioreg to query macOS HIDIdleTime."""
    try:
        output = subprocess.check_output(["ioreg", "-c", "IOHIDSystem"], timeout=2.0).decode("utf-8")
        for line in output.splitlines():
            if "HIDIdleTime" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    nanos = int(parts[1].strip())
                    return max(0.0, float(nanos) / 1e9)
    except Exception:
        pass
    return None


def get_linux_idle_seconds() -> float | None:
    """Uses xprintidle if installed on Linux X11 systems."""
    if shutil.which("xprintidle"):
        try:
            output = subprocess.check_output(["xprintidle"], timeout=2.0).decode("utf-8").strip()
            millis = int(output)
            return max(0.0, float(millis) / 1000.0)
        except Exception:
            pass
    return None


def user_idle_seconds() -> float | None:
    """
    Returns seconds since last user input (keyboard/mouse).
    Returns None if unsupported on the current platform/session (CPU-only idle fallback).
    """
    if sys.platform == "win32":
        return get_windows_idle_seconds()
    elif sys.platform == "darwin":
        return get_macos_idle_seconds()
    elif sys.platform.startswith("linux"):
        return get_linux_idle_seconds()
    return None


def system_cpu_excluding(pids: list[int] | None = None, interval: float = 0.2) -> float:
    """
    Computes system CPU utilization percentage.
    If specific worker pids are provided, subtracts their share where feasible.
    """
    try:
        total_cpu = psutil.cpu_percent(interval=interval)
        if not pids:
            return float(total_cpu)

        worker_cpu = 0.0
        for pid in pids:
            try:
                p = psutil.Process(pid)
                worker_cpu += p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        adjusted = max(0.0, total_cpu - worker_cpu)
        return float(adjusted)
    except Exception:
        return 0.0


def is_on_battery() -> bool:
    """
    Returns True if system is running on battery (not plugged into AC power).
    Returns False if plugged in or if no battery sensor exists (e.g. desktop).
    """
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return False
        # If power_plugged is False, device is discharging on battery
        return not bool(battery.power_plugged)
    except Exception:
        return False
