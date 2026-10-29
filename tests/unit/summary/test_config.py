from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from coola.summary import SummaryConfig

###################################
#     Tests for SummaryConfig     #
###################################


def test_summary_config_default() -> None:
    assert SummaryConfig() == SummaryConfig(
        max_depth=1, max_items=5, max_characters=1000, show_data=False
    )


def test_summary_config_repr() -> None:
    assert repr(SummaryConfig()) == (
        "SummaryConfig(max_depth=1, max_items=5, max_characters=1000, show_data=False)"
    )


def test_summary_config_custom() -> None:
    config = SummaryConfig(max_depth=3, max_items=2, max_characters=10, show_data=True)
    assert (config.max_depth, config.max_items, config.max_characters, config.show_data) == (
        3,
        2,
        10,
        True,
    )


@pytest.mark.parametrize("max_depth", [0, 1, 100])
def test_summary_config_max_depth_valid(max_depth: int) -> None:
    assert SummaryConfig(max_depth=max_depth).max_depth == max_depth


@pytest.mark.parametrize("name", ["max_items", "max_characters"])
@pytest.mark.parametrize("value", [0, 1, 100, None])
def test_summary_config_limit_valid(name: str, value: int | None) -> None:
    kwargs: dict[str, Any] = {name: value}
    assert getattr(SummaryConfig(**kwargs), name) == value


def test_summary_config_max_depth_none_raises() -> None:
    with pytest.raises(TypeError, match="max_depth"):
        SummaryConfig(max_depth=None)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("name", ["max_depth", "max_items", "max_characters"])
@pytest.mark.parametrize("value", [True, False, 1.5, "1"])
def test_summary_config_not_int_raises(name: str, value: object) -> None:
    kwargs: dict[str, Any] = {name: value}
    with pytest.raises(TypeError, match=name):
        SummaryConfig(**kwargs)


@pytest.mark.parametrize("name", ["max_depth", "max_items", "max_characters"])
def test_summary_config_negative_raises(name: str) -> None:
    kwargs: dict[str, Any] = {name: -1}
    with pytest.raises(ValueError, match=f"{name} must be greater than or equal to 0"):
        SummaryConfig(**kwargs)


@pytest.mark.parametrize("value", [0, 1, "True", None])
def test_summary_config_show_data_not_bool_raises(value: object) -> None:
    with pytest.raises(TypeError, match="show_data"):
        SummaryConfig(show_data=value)  # ty: ignore[invalid-argument-type]


def test_summary_config_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field 'max_depth'"):
        SummaryConfig().max_depth = 2  # ty: ignore[invalid-assignment]


def test_summary_config_is_hashable() -> None:
    assert hash(SummaryConfig()) == hash(SummaryConfig())


def test_summary_config_different_options_are_different() -> None:
    assert SummaryConfig(max_items=1) != SummaryConfig(max_items=2)
