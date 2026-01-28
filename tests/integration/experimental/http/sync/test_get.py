from __future__ import annotations

import pytest

from coola.experimental.http.sync import get_with_automatic_retry
from coola.testing.fixtures import httpx_available, httpx_not_available
from coola.utils.imports import is_httpx_available

if is_httpx_available():
    import httpx

# Use httpbin.org for real HTTP testing
HTTPBIN_URL = "https://httpbin.org"


##############################################
#     Tests for get_with_automatic_retry     #
##############################################


@httpx_available
def test_get_with_automatic_retry_successful_get_request() -> None:
    """Test successful GET request without retries."""
    response = get_with_automatic_retry(url=f"{HTTPBIN_URL}/get")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["url"] == "https://httpbin.org/get"


@httpx_available
def test_get_with_non_retryable_status_fails_immediately() -> None:
    """Test that 404 (non-retryable) fails immediately without
    retries."""
    with pytest.raises(httpx.HTTPStatusError):
        get_with_automatic_retry(url=f"{HTTPBIN_URL}/status/404")


@httpx_available
def test_get_with_automatic_retry_with_httpx() -> None:
    get_with_automatic_retry(url=f"{HTTPBIN_URL}/get")


@httpx_not_available
def test_get_with_automatic_retry_without_httpx() -> None:
    with pytest.raises(RuntimeError, match=r"'httpx' package is required but not installed."):
        get_with_automatic_retry(url=f"{HTTPBIN_URL}/get")
