r"""Unit tests for HttpRequestError exception."""

from __future__ import annotations

import pytest

from coola.experimental.http import HttpRequestError
from coola.testing.fixtures import httpx_available
from coola.utils.imports import is_httpx_available

if is_httpx_available():
    import httpx


@pytest.fixture
def basic_error() -> HttpRequestError:
    """Create a basic HttpRequestError with minimal parameters."""
    return HttpRequestError(
        method="GET", url="https://api.example.com/data", message="Request failed"
    )


@pytest.fixture
def error_with_status() -> HttpRequestError:
    """Create an HttpRequestError with status code."""
    return HttpRequestError(
        method="POST",
        url="https://api.example.com/submit",
        message="Server returned 404",
        status_code=404,
    )


@pytest.fixture
def mock_response() -> httpx.Response:
    """Create a mock httpx.Response object."""
    request = httpx.Request("GET", "https://api.example.com/data")
    return httpx.Response(status_code=500, request=request, json={"error": "Internal server error"})


######################################
#     Tests for HttpRequestError     #
######################################


def test_http_request_error_basic_initialization() -> None:
    """Test basic initialization with required parameters only."""
    error = HttpRequestError(method="GET", url="https://example.com", message="Test error")

    assert error.method == "GET"
    assert error.url == "https://example.com"
    assert str(error) == "Test error"
    assert error.status_code is None
    assert error.response is None


@httpx_available
def test_http_request_error_initialization_with_all_parameters(
    mock_response: httpx.Response,
) -> None:
    """Test initialization with all parameters provided."""
    cause = ConnectionError("Network failure")
    error = HttpRequestError(
        method="POST",
        url="https://api.example.com/endpoint",
        message="Complete error",
        status_code=503,
        response=mock_response,
        cause=cause,
    )

    assert error.method == "POST"
    assert error.url == "https://api.example.com/endpoint"
    assert str(error) == "Complete error"
    assert error.status_code == 503
    assert error.response == mock_response
    assert error.__cause__ == cause


def test_http_request_error_initialization_with_keyword_only_arguments() -> None:
    """Test that all arguments must be passed as keyword arguments."""
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        HttpRequestError("GET", "https://example.com", "Error message")


def test_http_request_error_method_attribute(basic_error: HttpRequestError) -> None:
    """Test that method attribute is correctly stored."""
    assert basic_error.method == "GET"


def test_http_request_error_url_attribute(basic_error: HttpRequestError) -> None:
    """Test that URL attribute is correctly stored."""
    assert basic_error.url == "https://api.example.com/data"


def test_http_request_error_status_code_attribute_when_none(basic_error: HttpRequestError) -> None:
    """Test status_code attribute when not provided."""
    assert basic_error.status_code is None


def test_http_request_error_status_code_attribute_when_provided(
    error_with_status: HttpRequestError,
) -> None:
    """Test status_code attribute when provided."""
    assert error_with_status.status_code == 404


def test_http_request_error_response_attribute_when_none(basic_error: HttpRequestError) -> None:
    """Test response attribute when not provided."""
    assert basic_error.response is None


@httpx_available
def test_http_request_error_response_attribute_when_provided(mock_response: httpx.Response) -> None:
    """Test response attribute when provided."""
    error = HttpRequestError(
        method="GET",
        url="https://api.example.com/data",
        message="Server error occurred",
        status_code=500,
        response=mock_response,
    )
    assert error.response is not None
    assert error.response.status_code == 500


def test_http_request_error_cause_is_set_when_provided() -> None:
    """Test that __cause__ is properly set when cause is provided."""
    error = HttpRequestError(
        method="DELETE",
        url="https://api.example.com/resource/123",
        message="Request validation failed",
        cause=ValueError("Invalid parameter"),
    )
    assert error.__cause__ is not None
    assert isinstance(error.__cause__, ValueError)
    assert str(error.__cause__) == "Invalid parameter"


def test_http_request_error_cause_is_none_when_not_provided(basic_error: HttpRequestError) -> None:
    """Test that __cause__ is None when cause is not provided."""
    assert basic_error.__cause__ is None


def test_http_request_error_exception_chaining_preserves_traceback() -> None:
    """Test that exception chaining preserves the original traceback."""
    original_error = RuntimeError("Original error")

    try:
        raise original_error
    except RuntimeError as e:
        chained_error = HttpRequestError(
            method="GET", url="https://example.com", message="Chained error", cause=e
        )

    assert chained_error.__cause__ is original_error


def test_http_request_error_is_instance_of_runtime_error(basic_error: HttpRequestError) -> None:
    """Test that HttpRequestError is a subclass of RuntimeError."""
    assert isinstance(basic_error, RuntimeError)


def test_http_request_error_is_instance_of_exception(basic_error: HttpRequestError) -> None:
    """Test that HttpRequestError is a subclass of Exception."""
    assert isinstance(basic_error, Exception)


def test_http_request_error_can_be_caught_as_runtime_error() -> None:
    """Test that HttpRequestError can be caught as RuntimeError."""
    with pytest.raises(RuntimeError):
        raise HttpRequestError(method="GET", url="https://example.com", message="Test")


def test_http_request_error_can_be_caught_as_specific_type() -> None:
    """Test that HttpRequestError can be caught by its specific type."""
    with pytest.raises(HttpRequestError):
        raise HttpRequestError(method="POST", url="https://example.com", message="Test")


def test_http_request_error_str_returns_message(basic_error: HttpRequestError) -> None:
    """Test that str() returns the error message."""
    assert str(basic_error) == "Request failed"


def test_http_request_error_repr_includes_key_attributes_without_status_code(
    basic_error: HttpRequestError,
) -> None:
    """Test that repr() includes method, url, and status_code."""
    assert (
        repr(basic_error)
        == "HttpRequestError(method='GET', url='https://api.example.com/data', status_code=None)"
    )


def test_http_request_error_repr_includes_key_attributes_with_status_code(
    error_with_status: HttpRequestError,
) -> None:
    """Test that repr() includes method, url, and status_code."""
    assert (
        repr(error_with_status)
        == "HttpRequestError(method='POST', url='https://api.example.com/submit', status_code=404)"
    )


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", ""])
def test_http_request_error_different_http_methods(method: str) -> None:
    """Test that various HTTP methods are stored correctly."""
    error = HttpRequestError(method=method, url="https://example.com", message="Test")
    assert error.method == method


@pytest.mark.parametrize("status_code", [200, 201, 400, 401, 403, 404, 500, 502, 503])
def test_http_request_error_different_status_codes(status_code: int) -> None:
    """Test that various HTTP status codes are stored correctly."""
    error = HttpRequestError(
        method="GET", url="https://example.com", message="Test", status_code=status_code
    )
    assert error.status_code == status_code


def test_http_request_error_empty_string_url() -> None:
    """Test initialization with empty string URL."""
    error = HttpRequestError(method="GET", url="", message="Test")
    assert error.url == ""


def test_http_request_error_empty_string_message() -> None:
    """Test initialization with empty string message."""
    error = HttpRequestError(method="GET", url="https://example.com", message="")
    assert str(error) == ""


def test_http_request_error_long_url() -> None:
    """Test initialization with a very long URL."""
    long_url = "https://example.com/" + "a" * 1000
    error = HttpRequestError(method="GET", url=long_url, message="Test")
    assert error.url == long_url


def test_http_request_error_unicode_in_message() -> None:
    """Test initialization with unicode characters in message."""
    error = HttpRequestError(
        method="GET", url="https://example.com", message="Error: 文字化け or émojis 🚀"
    )
    assert "文字化け" in str(error)
    assert "🚀" in str(error)


def test_http_request_error_special_characters_in_url() -> None:
    """Test initialization with special characters in URL."""
    url = "https://example.com/search?q=hello%20world&lang=en"
    error = HttpRequestError(method="GET", url=url, message="Test")
    assert error.url == url


@httpx_available
def test_http_request_error_with_real_httpx_response() -> None:
    """Test with an actual httpx.Response object."""
    request = httpx.Request("GET", "https://api.example.com/data")
    response = httpx.Response(
        status_code=404,
        request=request,
        headers={"content-type": "application/json"},
        json={"error": "Not found"},
    )

    error = HttpRequestError(
        method="GET",
        url="https://api.example.com/data",
        message="Resource not found",
        status_code=404,
        response=response,
    )

    assert error.response.status_code == 404
    assert error.response.json() == {"error": "Not found"}
    assert error.response.headers["content-type"] == "application/json"


@httpx_available
def test_http_request_error_with_httpx_exception_as_cause() -> None:
    """Test chaining with httpx exceptions."""
    httpx_error = httpx.ConnectError("Connection refused")

    error = HttpRequestError(
        method="POST",
        url="https://example.com/api",
        message="Failed to connect to server",
        cause=httpx_error,
    )

    assert isinstance(error.__cause__, httpx.ConnectError)
    assert "Connection refused" in str(error.__cause__)


def test_http_request_error_raising_and_catching() -> None:
    """Test that the exception can be raised and caught properly."""
    with pytest.raises(HttpRequestError) as exc_info:
        raise HttpRequestError(
            method="DELETE",
            url="https://example.com/resource",
            message="Deletion failed",
            status_code=403,
        )

    error = exc_info.value
    assert error.method == "DELETE"
    assert error.url == "https://example.com/resource"
    assert error.status_code == 403


def test_http_request_error_exception_message_in_traceback() -> None:
    """Test that the error message appears in the traceback."""
    with pytest.raises(HttpRequestError, match="Custom error message"):
        raise HttpRequestError(
            method="GET", url="https://example.com", message="Custom error message"
        )
