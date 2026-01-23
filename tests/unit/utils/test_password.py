r"""Unit tests for password utility functions."""

import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from coola.utils.password import MAX_PASSWORD_LENGTH, get_password


@pytest.fixture
def mock_interactive_terminal() -> Generator[None, None, None]:
    """Mock sys.stdin and sys.stderr as interactive terminals."""
    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch.object(sys.stderr, "isatty", return_value=True),
    ):
        yield


@pytest.fixture
def mock_non_interactive_terminal() -> Generator[None, None, None]:
    """Mock sys.stdin as non-interactive terminal."""
    with patch.object(sys.stdin, "isatty", return_value=False):
        yield


##################################
#     Tests for get_password     #
##################################


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_basic_success() -> None:
    """Test successful password retrieval with basic input."""
    with patch("getpass.getpass", return_value="secure_password123"):
        assert get_password() == "secure_password123"


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_with_username() -> None:
    """Test password prompt includes username when provided."""
    mock_getpass_func = MagicMock(return_value="mypassword")
    with patch("getpass.getpass", mock_getpass_func):
        assert get_password(user_name="alice") == "mypassword"
        mock_getpass_func.assert_called_once_with("Enter password for user alice: ")


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_without_username() -> None:
    """Test password prompt without username."""
    mock_getpass_func = MagicMock(return_value="mypassword")
    with patch("getpass.getpass", mock_getpass_func):
        assert get_password(user_name=None) == "mypassword"
        mock_getpass_func.assert_called_once_with("Enter password: ")


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_empty_raises_error() -> None:
    """Test that empty password raises ValueError."""
    with (
        patch("getpass.getpass", return_value=""),
        pytest.raises(ValueError, match="Password cannot be empty"),
    ):
        get_password()


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_exceeds_max_length() -> None:
    """Test that password exceeding max length raises ValueError."""

    long_password = "a" * (MAX_PASSWORD_LENGTH + 1)
    with (
        patch("getpass.getpass", return_value=long_password),
        pytest.raises(
            ValueError, match=f"Password exceeds maximum length of {MAX_PASSWORD_LENGTH}"
        ),
    ):
        get_password()


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_at_max_length() -> None:
    """Test that password at exactly max length is accepted."""

    max_length_password = "a" * MAX_PASSWORD_LENGTH
    with patch("getpass.getpass", return_value=max_length_password):
        password = get_password()
        assert password == max_length_password
        assert len(password) == MAX_PASSWORD_LENGTH


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_preserves_whitespace() -> None:
    """Test that whitespace in password is preserved."""
    password_with_spaces = "  my pass word  "  # noqa: S105
    with patch("getpass.getpass", return_value=password_with_spaces):
        assert get_password() == password_with_spaces


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_confirm_success() -> None:
    """Test successful password confirmation."""
    with patch("getpass.getpass", side_effect=["mypassword", "mypassword"]):
        assert get_password(confirm=True) == "mypassword"


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_confirm_mismatch() -> None:
    """Test that mismatched confirmation raises ValueError."""
    with (
        patch("getpass.getpass", side_effect=["password1", "password2"]),
        pytest.raises(ValueError, match="Passwords do not match"),
    ):
        get_password(confirm=True)


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_confirm_prompt_called_twice() -> None:
    """Test that confirmation mode calls getpass twice."""
    mock_getpass_func = MagicMock(side_effect=["mypassword", "mypassword"])
    with patch("getpass.getpass", mock_getpass_func):
        get_password(confirm=True)
        assert mock_getpass_func.call_count == 2
        calls = mock_getpass_func.call_args_list
        assert calls[0][0][0] == "Enter password: "
        assert calls[1][0][0] == "Confirm password: "


@pytest.mark.usefixtures("mock_non_interactive_terminal")
def test_get_password_non_interactive_terminal() -> None:
    """Test that non-interactive terminal raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Password input requires an interactive terminal"):
        get_password()


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_stdin_not_tty() -> None:
    """Test that non-tty stdin raises RuntimeError."""
    with (
        patch.object(sys.stdin, "isatty", return_value=False),
        pytest.raises(RuntimeError, match="Password input requires an interactive terminal"),
    ):
        get_password()


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_stderr_not_tty() -> None:
    """Test that non-tty stderr raises RuntimeError."""
    with (
        patch.object(sys.stderr, "isatty", return_value=False),
        pytest.raises(RuntimeError, match="Password input requires an interactive terminal"),
    ):
        get_password()


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_keyboard_interrupt() -> None:
    """Test that KeyboardInterrupt is propagated."""
    with (
        patch("getpass.getpass", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        get_password()


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_eof_error() -> None:
    """Test that EOFError is propagated."""
    with patch("getpass.getpass", side_effect=EOFError), pytest.raises(EOFError):
        get_password()


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_special_characters() -> None:
    """Test password with special characters is accepted."""
    special_password = "!@#$%^&*()_+-={}[]|:\";'<>?,./"  # noqa: S105
    with patch("getpass.getpass", return_value=special_password):
        assert get_password() == special_password


@pytest.mark.usefixtures("mock_interactive_terminal")
def test_get_password_unicode_characters() -> None:
    """Test password with unicode characters is accepted."""
    unicode_password = "пароль密码🔒"  # noqa: S105
    with patch("getpass.getpass", return_value=unicode_password):
        assert get_password() == unicode_password
