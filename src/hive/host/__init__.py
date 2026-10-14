"""
Hive Host server module.
Manages sessions, rounds, persistence, robust aggregation, public FL protocol,
admin dashboard endpoints, and reachability tunnels.
"""

from hive.host.app import create_host_apps, run_host
from hive.host.db import HostDatabase
from hive.host.metrics import get_contribution_metrics
from hive.host.orchestrator import Orchestrator
from hive.host.privacy import get_privacy_indicators
from hive.host.state import HostState
from hive.host.tunnel import TunnelManager, run_doctor

__all__ = [
    "HostDatabase",
    "HostState",
    "Orchestrator",
    "create_host_apps",
    "run_host",
    "TunnelManager",
    "run_doctor",
    "get_contribution_metrics",
    "get_privacy_indicators",
]