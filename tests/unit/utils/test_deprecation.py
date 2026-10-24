from __future__ import annotations

import inspect
from typing import Any

import pytest

from coola.utils.deprecation import deprecate_positional_options

###############################################
#     Tests for deprecate_positional_options  #
###############################################


@deprecate_positional_options("scale", "offset")
def affine(value: int, *, scale: int = 2, offset: int = 0) -> int:
    r"""Scale a value and add an offset."""
    return value * scale + offset


@deprecate_positional_options("scale")
def generate(value: int, *, scale: int = 2) -> Any:
    yield value * scale


def test_deprecate_positional_options_keywords_do_not_warn() -> None:
    assert affine(3, scale=4, offset=1) == 13


def test_deprecate_positional_options_no_option() -> None:
    assert affine(3) == 6


def test_deprecate_positional_options_one_positional_option_warns() -> None:
    with pytest.warns(DeprecationWarning, match="`scale` to `affine` as positional arguments"):
        affine(3, 4)


def test_deprecate_positional_options_two_positional_options_warn() -> None:
    with pytest.warns(DeprecationWarning, match="`scale`, `offset` to `affine`"):
        affine(3, 4, 1)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deprecate_positional_options_one_positional_option_result() -> None:
    assert affine(3, 4) == 12


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deprecate_positional_options_two_positional_options_result() -> None:
    assert affine(3, 4, 1) == 13


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deprecate_positional_options_positional_and_keyword_options_result() -> None:
    assert affine(3, 4, offset=1) == 13


def test_deprecate_positional_options_warning_points_to_the_caller() -> None:
    with pytest.warns(DeprecationWarning, match="is deprecated") as record:
        affine(3, 4)
    assert record[0].filename == __file__


def test_deprecate_positional_options_too_many_positional_arguments_raises() -> None:
    with pytest.raises(TypeError, match=r"affine\(\) takes at most 3 positional arguments but 4"):
        affine(3, 4, 1, 0)


def test_deprecate_positional_options_option_given_twice_raises() -> None:
    with (
        pytest.warns(DeprecationWarning, match="is deprecated"),
        pytest.raises(TypeError, match="multiple values for argument 'scale'"),
    ):
        affine(3, 4, scale=5)


def test_deprecate_positional_options_generator_warns_when_called() -> None:
    with pytest.warns(DeprecationWarning, match="`scale` to `generate`"):
        generate(3, 4)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deprecate_positional_options_generator_result() -> None:
    assert list(generate(3, 4)) == [12]


def test_deprecate_positional_options_keeps_the_name() -> None:
    assert affine.__name__ == "affine"


def test_deprecate_positional_options_keeps_the_docstring() -> None:
    assert affine.__doc__ == "Scale a value and add an offset."


def test_deprecate_positional_options_keeps_the_keyword_only_signature() -> None:
    assert inspect.signature(affine).parameters["scale"].kind is inspect.Parameter.KEYWORD_ONLY
