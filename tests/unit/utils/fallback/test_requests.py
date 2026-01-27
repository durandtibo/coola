from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.requests import HTTPAdapter, requests


def test_requests() -> None:
    isinstance(requests, ModuleType)


def test_http_adapter() -> None:
    with pytest.raises(RuntimeError, match=r"'requests' package is required but not installed."):
        HTTPAdapter()


def test_http_adapter_with_args() -> None:
    with pytest.raises(RuntimeError, match=r"'requests' package is required but not installed."):
        HTTPAdapter(max_retries=3)
