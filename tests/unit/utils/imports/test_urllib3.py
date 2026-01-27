from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_urllib3,
    is_urllib3_available,
    raise_urllib3_missing_error,
    urllib3_available,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_urllib3_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


###################################
#     Tests for check_urllib3     #
###################################


def test_check_urllib3_with_package() -> None:
    with patch("coola.utils.imports.urllib3.is_urllib3_available", lambda: True):
        check_urllib3()


def test_check_urllib3_without_package() -> None:
    with (
        patch("coola.utils.imports.urllib3.is_urllib3_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'urllib3' package is required but not installed."),
    ):
        check_urllib3()


###########################################
#     Testes for is_urllib3_available     #
###########################################


def test_is_urllib3_available() -> None:
    assert isinstance(is_urllib3_available(), bool)


########################################
#     Testes for urllib3_available     #
########################################


def test_urllib3_available_with_package() -> None:
    with patch("coola.utils.imports.urllib3.is_urllib3_available", lambda: True):
        fn = urllib3_available(my_function)
        assert fn(2) == 44


def test_urllib3_available_without_package() -> None:
    with patch("coola.utils.imports.urllib3.is_urllib3_available", lambda: False):
        fn = urllib3_available(my_function)
        assert fn(2) is None


def test_urllib3_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.urllib3.is_urllib3_available", lambda: True):

        @urllib3_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_urllib3_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.urllib3.is_urllib3_available", lambda: False):

        @urllib3_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


##################################################
#     Testes for raise_urllib3_missing_error     #
##################################################


def test_raise_urllib3_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'urllib3' package is required but not installed."):
        raise_urllib3_missing_error()
