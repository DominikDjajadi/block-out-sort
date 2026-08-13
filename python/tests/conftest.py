"""Shared test configuration: make the package importable without installation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

FIXTURES_DIR = REPO_ROOT / "fixtures"
CONFORMANCE_DIR = FIXTURES_DIR / "conformance"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def conformance_dir() -> Path:
    return CONFORMANCE_DIR
