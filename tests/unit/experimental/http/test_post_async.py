from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from coola.experimental.http import (
    DEFAULT_TIMEOUT,
    RETRY_STATUS_CODES,
    HttpRequestError,
    post_with_automatic_retry_async,
)
from coola.utils.imports import is_httpx_available

if is_httpx_available():
    import httpx


@pytest.fixture
def mock_client() -> httpx.AsyncClient:
    """Create a mock AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def success_response():
    """Create a successful HTTP response."""
    response = Mock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {"success": True}
    return response


@pytest.fixture
def server_error_response():
    """Create a server error HTTP response."""
    response = Mock(spec=httpx.Response)
    response.status_code = 500
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server Error", request=Mock(), response=response
    )
    return response


@pytest.mark.asyncio
async def test_successful_request_first_attempt(mock_client, success_response) -> None:
    """Test successful request on first attempt."""
    mock_client.post = AsyncMock(return_value=success_response)

    response = await post_with_automatic_retry_async(
        "https://api.example.com/test",
        client=mock_client,
    )

    assert response.status_code == 200
    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_successful_request_after_retry(
    mock_client, server_error_response, success_response
) -> None:
    """Test successful request after one retry."""
    mock_client.post = AsyncMock(side_effect=[server_error_response, success_response])

    response = await post_with_automatic_retry_async(
        "https://api.example.com/test",
        client=mock_client,
        max_retries=2,
    )

    assert response.status_code == 200
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_max_retries_exhausted(mock_client, server_error_response) -> None:
    """Test that HttpRequestError is raised when max retries
    exhausted."""
    mock_client.post = AsyncMock(return_value=server_error_response)

    with pytest.raises(HttpRequestError) as exc_info:
        await post_with_automatic_retry_async(
            "https://api.example.com/test",
            client=mock_client,
            max_retries=2,
            backoff_factor=0,  # No delay for faster tests
        )

    assert "failed with status 500" in str(exc_info.value)
    assert mock_client.post.call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_non_retryable_status_code(mock_client) -> None:
    """Test that non-retryable status codes raise immediately."""
    response = Mock(spec=httpx.Response)
    response.status_code = 404
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=Mock(), response=response
    )
    mock_client.post = AsyncMock(return_value=response)

    with pytest.raises(httpx.HTTPStatusError):
        await post_with_automatic_retry_async(
            "https://api.example.com/test",
            client=mock_client,
            max_retries=3,
        )

    assert mock_client.post.call_count == 1  # No retries for 404


@pytest.mark.asyncio
async def test_timeout_exception(mock_client) -> None:
    """Test handling of timeout exceptions."""
    mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

    with pytest.raises(HttpRequestError) as exc_info:
        await post_with_automatic_retry_async(
            "https://api.example.com/test",
            client=mock_client,
            max_retries=1,
            backoff_factor=0,
        )

    assert "timed out" in str(exc_info.value)
    assert mock_client.post.call_count == 2  # Initial + 1 retry


@pytest.mark.asyncio
async def test_request_error(mock_client) -> None:
    """Test handling of request errors."""
    mock_client.post = AsyncMock(side_effect=httpx.RequestError("Connection failed"))

    with pytest.raises(HttpRequestError) as exc_info:
        await post_with_automatic_retry_async(
            "https://api.example.com/test",
            client=mock_client,
            max_retries=2,
            backoff_factor=0,
        )

    assert "failed after 3 attempts" in str(exc_info.value)
    assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_custom_status_forcelist(mock_client) -> None:
    """Test custom status codes for retry."""
    response = Mock(spec=httpx.Response)
    response.status_code = 429
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Too Many Requests", request=Mock(), response=response
    )
    mock_client.post = AsyncMock(return_value=response)

    # 429 not in custom list, should fail immediately
    with pytest.raises(httpx.HTTPStatusError):
        await post_with_automatic_retry_async(
            "https://api.example.com/test",
            client=mock_client,
            status_forcelist=(500, 502, 503),
            max_retries=3,
        )

    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_exponential_backoff() -> None:
    """Test exponential backoff timing."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    error_response = Mock(spec=httpx.Response)
    error_response.status_code = 503
    mock_client.post = AsyncMock(return_value=error_response)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(HttpRequestError):
            await post_with_automatic_retry_async(
                "https://api.example.com/test",
                client=mock_client,
                max_retries=3,
                backoff_factor=1.0,
            )

        # Check exponential backoff: 1*2^0, 1*2^1, 1*2^2
        assert mock_sleep.call_count == 3
        sleep_times = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_times == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_client_auto_creation_and_cleanup() -> None:
    """Test that client is created and cleaned up when not provided."""
    success_response = Mock(spec=httpx.Response)
    success_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=success_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        await post_with_automatic_retry_async("https://api.example.com/test")

        mock_client_class.assert_called_once_with(timeout=DEFAULT_TIMEOUT)
        mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_client_not_closed_when_provided(mock_client, success_response) -> None:
    """Test that provided client is not closed."""
    mock_client.post = AsyncMock(return_value=success_response)

    await post_with_automatic_retry_async(
        "https://api.example.com/test",
        client=mock_client,
    )

    mock_client.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_kwargs_passed_to_post(mock_client, success_response) -> None:
    """Test that additional kwargs are passed to post method."""
    mock_client.post = AsyncMock(return_value=success_response)

    await post_with_automatic_retry_async(
        "https://api.example.com/test",
        client=mock_client,
        json={"key": "value"},
        headers={"Authorization": "Bearer token"},
    )

    mock_client.post.assert_called_once_with(
        url="https://api.example.com/test",
        json={"key": "value"},
        headers={"Authorization": "Bearer token"},
    )


@pytest.mark.asyncio
async def test_negative_max_retries() -> None:
    """Test that negative max_retries raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        await post_with_automatic_retry_async(
            "https://api.example.com/test",
            max_retries=-1,
        )

    assert "max_retries must be >= 0" in str(exc_info.value)


@pytest.mark.asyncio
async def test_negative_backoff_factor() -> None:
    """Test that negative backoff_factor raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        await post_with_automatic_retry_async(
            "https://api.example.com/test",
            backoff_factor=-1.0,
        )

    assert "backoff_factor must be >= 0" in str(exc_info.value)


@pytest.mark.asyncio
async def test_zero_max_retries(mock_client, server_error_response) -> None:
    """Test behavior with zero max retries."""
    mock_client.post = AsyncMock(return_value=server_error_response)

    with pytest.raises(HttpRequestError):
        await post_with_automatic_retry_async(
            "https://api.example.com/test",
            client=mock_client,
            max_retries=0,
            backoff_factor=0,
        )

    assert mock_client.post.call_count == 1  # Only initial attempt


@pytest.mark.asyncio
async def test_all_retry_status_codes(mock_client, success_response) -> None:
    """Test that all default retry status codes trigger retries."""
    for status_code in RETRY_STATUS_CODES:
        mock_client.reset_mock()
        error_response = Mock(spec=httpx.Response)
        error_response.status_code = status_code

        mock_client.post = AsyncMock(side_effect=[error_response, success_response])

        response = await post_with_automatic_retry_async(
            "https://api.example.com/test",
            client=mock_client,
            max_retries=1,
            backoff_factor=0,
        )

        assert response.status_code == 200
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_success_status_codes(mock_client: httpx.AsyncClient) -> None:
    """Test that various success status codes are accepted."""
    for status_code in [200, 201, 204]:
        mock_client.reset_mock()
        response = Mock(spec=httpx.Response)
        response.status_code = status_code

        mock_client.post = AsyncMock(return_value=response)

        result = await post_with_automatic_retry_async(
            "https://api.example.com/test",
            client=mock_client,
        )

        assert result.status_code == status_code
        assert mock_client.post.call_count == 1
