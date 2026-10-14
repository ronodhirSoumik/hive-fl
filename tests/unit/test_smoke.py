"""
Smoke test verifying importability of all Hive packages.
"""

import importlib
import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "hive",
        "hive.core",
        "hive.core.config",
        "hive.core.logging",
        "hive.cli.main",
        "hive.package",
        "hive.host",
        "hive.host.aggregation",
        "hive.client",
        "hive.client.sandbox",
        "hive.client.worker",
        "hive.demo",
    ],
)
def test_package_import(module_name: str) -> None:
    """Verifies that each top-level Hive package can be cleanly imported."""
    mod = importlib.import_module(module_name)
    assert mod is not None