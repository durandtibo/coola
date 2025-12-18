from __future__ import annotations

import pytest

from coola.utils.fallback.packaging import Version


def test_version() -> None:
    with pytest.raises(RuntimeError, match=r"'packaging' package is required but not installed."):
        Version()


def test_version_with_args() -> None:
    with pytest.raises(RuntimeError, match=r"'packaging' package is required but not installed."):
        Version("1.0")
