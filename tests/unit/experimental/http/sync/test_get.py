from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, call, patch

import pytest

from coola.experimental.http import HttpRequestError
from coola.experimental.http.sync import get_with_automatic_retry
from coola.testing.fixtures import httpx_available
from coola.utils.imports import is_httpx_available

if TYPE_CHECKING:
    from collections.abc import Generator

if is_httpx_available():
    import httpx


@pytest.fixture
def mock_sleep() -> Generator[Mock, None, None]:
    """Patch time.sleep to make tests run faster."""
    with patch("time.sleep", return_value=None) as mock:
        yield mock


##############################################
#     Tests for get_with_automatic_retry     #
##############################################


@httpx_available
def test_get_with_automatic_retry_successful_get_request(mock_sleep: Mock) -> None:
    """Test successful GET request on first attempt."""
    mock_response = Mock(spec=httpx.Response, status_code=200)

    with patch("httpx.Client.get", return_value=mock_response):
        response = get_with_automatic_retry("https://api.example.com/data")

    mock_sleep.assert_not_called()
    assert response.status_code == 200


@httpx_available
def test_get_with_automatic_retry_successful_get_with_custom_client(mock_sleep: Mock) -> None:
    """Test successful GET request with custom client."""
    mock_response = Mock(spec=httpx.Response, status_code=201)
    mock_client = Mock(spec=httpx.Client, get=Mock(return_value=mock_response))

    response = get_with_automatic_retry("https://api.example.com/data", client=mock_client)

    mock_client.get.assert_called_once_with(url="https://api.example.com/data")
    mock_sleep.assert_not_called()
    assert response.status_code == 201


@httpx_available
def test_get_with_automatic_retry_get_with_json_payload(mock_sleep: Mock) -> None:
    """Test GET request with JSON data."""
    mock_response = Mock(spec=httpx.Response, status_code=200)

    with patch("httpx.Client.get", return_value=mock_response) as mock_get:
        response = get_with_automatic_retry("https://api.example.com/data", json={"key": "value"})

    mock_get.assert_called_once_with(url="https://api.example.com/data", json={"key": "value"})
    mock_sleep.assert_not_called()
    assert response.status_code == 200


@httpx_available
def test_get_with_automatic_retry_retry_on_500_status(mock_sleep: Mock) -> None:
    """Test retry logic for 500 status code."""
    mock_response_fail = Mock(spec=httpx.Response, status_code=500)
    mock_response_success = Mock(spec=httpx.Response, status_code=200)

    with patch("httpx.Client.get", side_effect=[mock_response_fail, mock_response_success]):
        response = get_with_automatic_retry("https://api.example.com/data")

    assert response.status_code == 200
    mock_sleep.assert_called_once_with(0.3)


@httpx_available
def test_get_with_automatic_retry_retry_on_503_status(mock_sleep: Mock) -> None:
    """Test retry logic for 503 status code."""
    mock_response_fail = Mock(spec=httpx.Response, status_code=503)
    mock_response_success = Mock(spec=httpx.Response, status_code=200)

    with patch("httpx.Client.get", side_effect=[mock_response_fail, mock_response_success]):
        response = get_with_automatic_retry("https://api.example.com/data")

    assert response.status_code == 200
    mock_sleep.assert_called_once_with(0.3)


@httpx_available
def test_get_with_automatic_retry_max_retries_exceeded(mock_sleep: Mock) -> None:
    """Test that HttpRequestError is raised when max retries
    exceeded."""
    mock_response = Mock(spec=httpx.Response, status_code=503)

    with (
        patch("httpx.Client.get", return_value=mock_response),
        pytest.raises(HttpRequestError) as exc_info,
    ):
        get_with_automatic_retry("https://api.example.com/data", max_retries=2)

    assert exc_info.value.status_code == 503
    assert "failed with status 503 after 3 attempts" in str(exc_info.value)
    assert mock_sleep.call_args_list == [call(0.3), call(0.6)]


@httpx_available
def test_get_with_automatic_retry_non_retryable_status_code(mock_sleep: Mock) -> None:
    """Test that 404 status code is not retried."""
    mock_response = Mock(spec=httpx.Response, status_code=404)
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=Mock(), response=mock_response
    )

    with (
        patch("httpx.Client.get", return_value=mock_response),
        pytest.raises(httpx.HTTPStatusError, match="Not Found"),
    ):
        get_with_automatic_retry("https://api.example.com/data")
    mock_sleep.assert_not_called()


@httpx_available
def test_get_with_automatic_retry_exponential_backoff(mock_sleep: Mock) -> None:
    """Test exponential backoff timing."""
    mock_response_fail = Mock(spec=httpx.Response, status_code=503)
    mock_response_success = Mock(spec=httpx.Response, status_code=200)

    with patch(
        "httpx.Client.get",
        side_effect=[mock_response_fail, mock_response_fail, mock_response_success],
    ):
        get_with_automatic_retry("https://api.example.com/data", max_retries=3, backoff_factor=2.0)

    # Should have slept twice (after 1st and 2nd failures)
    assert mock_sleep.call_args_list == [call(2.0), call(4.0)]


@httpx_available
def test_get_with_automatic_retry_timeout_exception(mock_sleep: Mock) -> None:
    """Test handling of timeout exception."""
    with (
        patch("httpx.Client.get", side_effect=httpx.TimeoutException("Request timeout")),
        pytest.raises(
            HttpRequestError,
            match=r"GET request to https://api.example.com/data timed out \(1 attempts\)",
        ),
    ):
        get_with_automatic_retry("https://api.example.com/data", max_retries=0)
    mock_sleep.assert_not_called()


@httpx_available
def test_get_with_automatic_retry_timeout_exception_with_retries(mock_sleep: Mock) -> None:
    """Test timeout exception with retries."""
    with (
        patch("httpx.Client.get", side_effect=httpx.TimeoutException("Request timeout")),
        pytest.raises(
            HttpRequestError,
            match=r"GET request to https://api.example.com/data timed out \(3 attempts\)",
        ),
    ):
        get_with_automatic_retry("https://api.example.com/data", max_retries=2)
    assert mock_sleep.call_args_list == [call(0.3), call(0.6)]


@httpx_available
def test_get_with_automatic_retry_request_error(mock_sleep: Mock) -> None:
    """Test handling of general request errors."""
    with (
        patch("httpx.Client.get", side_effect=httpx.RequestError("Connection failed")),
        pytest.raises(HttpRequestError, match="failed after 1 attempts"),
    ):
        get_with_automatic_retry("https://api.example.com/data", max_retries=0)
    mock_sleep.assert_not_called()


@httpx_available
def test_get_with_automatic_retry_request_error_with_retries(mock_sleep: Mock) -> None:
    """Test handling of general request errors."""
    with (
        patch("httpx.Client.get", side_effect=httpx.RequestError("Connection failed")),
        pytest.raises(HttpRequestError, match="failed after 3 attempts"),
    ):
        get_with_automatic_retry("https://api.example.com/data", max_retries=2)
    assert mock_sleep.call_args_list == [call(0.3), call(0.6)]


@httpx_available
def test_get_with_automatic_retry_negative_max_retries() -> None:
    """Test that negative max_retries raises ValueError."""
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        get_with_automatic_retry("https://api.example.com/data", max_retries=-1)


@httpx_available
def test_get_with_automatic_retry_negative_backoff_factor() -> None:
    """Test that negative backoff_factor raises ValueError."""
    with pytest.raises(ValueError, match="backoff_factor must be >= 0"):
        get_with_automatic_retry("https://api.example.com/data", backoff_factor=-1.0)


@httpx_available
def test_get_with_automatic_retry_zero_max_retries(mock_sleep: Mock) -> None:
    """Test with zero retries - should only try once."""
    mock_response = Mock(spec=httpx.Response, status_code=503)

    with (
        patch("httpx.Client.get", return_value=mock_response),
        pytest.raises(HttpRequestError, match="after 1 attempts"),
    ):
        get_with_automatic_retry("https://api.example.com/data", max_retries=0)
    mock_sleep.assert_not_called()


@httpx_available
def test_get_with_automatic_retry_custom_status_forcelist(mock_sleep: Mock) -> None:
    """Test custom status codes for retry."""
    mock_response = Mock(spec=httpx.Response, status_code=404)
    mock_response_success = Mock(spec=httpx.Response, status_code=200)

    with patch("httpx.Client.get", side_effect=[mock_response, mock_response_success]):
        response = get_with_automatic_retry("https://api.example.com/data", status_forcelist=(404,))

    assert response.status_code == 200
    mock_sleep.assert_called_once_with(0.3)


@httpx_available
def test_get_with_automatic_retry_client_close_when_owns_client() -> None:
    """Test that client is closed when created internally."""
    mock_response = Mock(spec=httpx.Response, status_code=200)
    mock_client = Mock(spec=httpx.Client, get=Mock(return_value=mock_response))

    with patch("httpx.Client", return_value=mock_client):
        get_with_automatic_retry("https://api.example.com/data")

    mock_client.close.assert_called_once()


@httpx_available
def test_get_with_automatic_retry_client_not_closed_when_provided() -> None:
    """Test that external client is not closed."""
    mock_response = Mock(spec=httpx.Response, status_code=200)
    mock_client = Mock(spec=httpx.Client, get=Mock(return_value=mock_response))

    get_with_automatic_retry("https://api.example.com/data", client=mock_client)

    mock_client.close.assert_not_called()


@httpx_available
def test_get_with_automatic_retry_custom_timeout(mock_sleep: Mock) -> None:
    """Test custom timeout parameter."""
    mock_response = Mock(spec=httpx.Response, status_code=200)
    mock_client = Mock(spec=httpx.Client, get=Mock(return_value=mock_response))

    with patch("httpx.Client") as mock_client_class:
        mock_client_class.return_value = mock_client
        get_with_automatic_retry("https://api.example.com/data", timeout=30.0)

    mock_client_class.assert_called_once_with(timeout=30.0)
    mock_sleep.assert_not_called()


@httpx_available
def test_get_with_automatic_retry_all_retries_with_429(mock_sleep: Mock) -> None:
    """Test retry behavior with 429 Too Many Requests."""
    mock_response = Mock(spec=httpx.Response, status_code=429)

    with (
        patch("httpx.Client.get", return_value=mock_response),
        pytest.raises(HttpRequestError) as exc_info,
    ):
        get_with_automatic_retry("https://api.example.com/data", max_retries=1)

    assert exc_info.value.status_code == 429
    assert "failed with status 429 after 2 attempts" in str(exc_info.value)
    assert mock_sleep.call_args_list == [call(0.3)]


def test_get_with_automatic_retry_without_httpx() -> None:
    with (
        patch("coola.utils.imports.httpx.is_httpx_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'httpx' package is required but not installed."),
    ):
        get_with_automatic_retry(url="https://api.example.com/data")
