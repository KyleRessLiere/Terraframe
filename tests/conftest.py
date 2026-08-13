"""Shared pytest configuration.

Tests marked ``@pytest.mark.network`` hit live tile servers and are skipped by
default. Opt in with ``pytest --network`` (or ``-m network --network`` to run
only those).
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="run tests that require live network access",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--network"):
        return
    skip = pytest.mark.skip(reason="needs --network to run (live tile download)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
