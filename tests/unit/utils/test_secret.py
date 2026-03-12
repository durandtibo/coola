import pytest

from coola.utils.secret import mask_secret


@pytest.fixture
def secret() -> str:
    return "abcdefghijklmnopqrstuvwxyz"


#################################
#     Tests for mask_secret     #
#################################


def test_mask_secret_default(secret: str) -> None:
    assert mask_secret(secret) == "abc*******************wxyz"


def test_mask_secret_show_first_zero(secret: str) -> None:
    assert mask_secret(secret, show_first=0) == "**********************wxyz"


def test_mask_secret_show_last_zero(secret: str) -> None:
    assert mask_secret(secret, show_last=0) == "abc***********************"


def test_mask_secret_show_first_and_last_zero(secret: str) -> None:
    assert mask_secret(secret, show_first=0, show_last=0) == "**************************"


def test_mask_secret_short_string_masked_fully() -> None:
    assert mask_secret("ab") == "**"


def test_mask_secret_exact_length_boundary() -> None:
    assert mask_secret("abcdefg") == "*******"


def test_mask_secret_custom_show_first_and_last(secret: str) -> None:
    assert mask_secret(secret, show_first=5, show_last=5) == "abcde****************vwxyz"


def test_mask_secret_single_char() -> None:
    assert mask_secret("x") == "*"


def test_mask_secret_empty_string() -> None:
    assert mask_secret("") == ""
