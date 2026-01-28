from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.httpx import httpx


def test_httpx_is_module_type() -> None:
    assert isinstance(httpx, ModuleType)


def test_httpx_module_name() -> None:
    assert httpx.__name__ == "httpx"


def test_httpx_response_exists() -> None:
    assert hasattr(httpx, "Response")


def test_httpx_response_is_class() -> None:
    assert isinstance(httpx.Response, type)


def test_httpx_response_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'httpx' package is required but not installed."):
        httpx.Response()
