from __future__ import annotations

import logging

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, SetSameValuesHandler, TrueHandler


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


def test_set_same_values_handler_equal_true() -> None:
    assert SetSameValuesHandler().equal(SetSameValuesHandler())


def test_set_same_values_handler_equal_false_different_type() -> None:
    assert not SetSameValuesHandler().equal(TrueHandler())


def test_set_same_values_handler_equal_false_different_next() -> None:
    assert not SetSameValuesHandler(TrueHandler()).equal(SetSameValuesHandler())


def test_set_same_values_handler_str() -> None:
    assert str(SetSameValuesHandler()) == "SetSameValuesHandler()"


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (set(), set()),
        ({1}, {1}),
        ({1, 2, 3}, {3, 2, 1}),
        ({"a", "b"}, {"b", "a"}),
        ({None, b"x", "y", 1}, {None, b"x", "y", 1}),
        (frozenset({1, 2}), frozenset({2, 1})),
        ({(1, 2), (3, 4)}, {(3, 4), (1, 2)}),
        ({1.5, 2.5}, {2.5, 1.5}),
    ],
)
def test_set_same_values_handler_handle_true(
    config: EqualityConfig, actual: set, expected: set
) -> None:
    assert SetSameValuesHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({1}, {2}),
        ({1, 2}, {1, 3}),
        ({"a"}, {"b"}),
        ({1.5}, {2.5}),
        ({(1, 2)}, {(1, 3)}),
        ({1}, {True}),
        ({1, False}, {True, 0}),
        ({1}, {1.0}),
    ],
)
def test_set_same_values_handler_handle_false(
    config: EqualityConfig, actual: set, expected: set
) -> None:
    assert not SetSameValuesHandler(next_handler=TrueHandler()).handle(actual, expected, config)


def test_set_same_values_handler_handle_different_lengths(config: EqualityConfig) -> None:
    assert not SetSameValuesHandler(next_handler=TrueHandler()).handle({1, 2}, {1}, config)


def test_set_same_values_handler_handle_calls_next(config: EqualityConfig) -> None:
    assert not SetSameValuesHandler(next_handler=FalseHandler()).handle({1}, {1}, config)


def test_set_same_values_handler_handle_nan_different_by_default(config: EqualityConfig) -> None:
    assert not SetSameValuesHandler(next_handler=TrueHandler()).handle(
        {1.0, float("nan")}, {1.0, float("nan")}, config
    )


def test_set_same_values_handler_handle_equal_nan() -> None:
    config = EqualityConfig(equal_nan=True)
    assert SetSameValuesHandler(next_handler=TrueHandler()).handle(
        {1.0, float("nan")}, {1.0, float("nan")}, config
    )


def test_set_same_values_handler_handle_equal_nan_in_a_tuple() -> None:
    config = EqualityConfig(equal_nan=True)
    assert SetSameValuesHandler(next_handler=TrueHandler()).handle(
        {(float("nan"), 1)}, {(float("nan"), 1)}, config
    )


def test_set_same_values_handler_handle_tolerance_is_not_applied() -> None:
    config = EqualityConfig(atol=1e-6)
    assert not SetSameValuesHandler(next_handler=TrueHandler()).handle({1.0}, {1.0 + 1e-9}, config)


def test_set_same_values_handler_handle_equal_nan_in_a_frozenset() -> None:
    config = EqualityConfig(equal_nan=True)
    assert SetSameValuesHandler(next_handler=TrueHandler()).handle(
        {frozenset({float("nan")})}, {frozenset({float("nan")})}, config
    )


def test_set_same_values_handler_handle_equal_nan_in_a_nested_tuple() -> None:
    config = EqualityConfig(equal_nan=True)
    assert SetSameValuesHandler(next_handler=TrueHandler()).handle(
        {((float("nan"), 1), 2)}, {((float("nan"), 1), 2)}, config
    )


def test_set_same_values_handler_handle_equal_nan_matches_each_nan_once() -> None:
    config = EqualityConfig(equal_nan=True)
    assert not SetSameValuesHandler(next_handler=TrueHandler()).handle(
        {float("nan"), -float("nan")}, {float("nan"), 1.0}, config
    )


def test_set_same_values_handler_handle_tuple_and_frozenset_differ() -> None:
    config = EqualityConfig()
    assert not SetSameValuesHandler(next_handler=TrueHandler()).handle(
        {(1, 2)}, frozenset({frozenset({1, 2})}), config
    )


def test_set_same_values_handler_handle_does_not_change_the_inputs(config: EqualityConfig) -> None:
    actual, expected = {1.5, 2.5}, {2.5, 1.5}
    SetSameValuesHandler(next_handler=TrueHandler()).handle(actual, expected, config)
    assert (actual, expected) == ({1.5, 2.5}, {2.5, 1.5})


def test_set_same_values_handler_handle_show_difference_logs_the_element(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = EqualityConfig(show_difference=True)
    with caplog.at_level(logging.INFO):
        SetSameValuesHandler(next_handler=TrueHandler()).handle({1.5}, {2.5}, config)
    assert caplog.messages == [
        "sets have different elements: 1.5 of actual has no match in expected"
    ]


def test_set_same_values_handler_handle_show_difference_reports_the_element_as_the_path() -> None:
    reports: list[tuple[str, tuple]] = []
    config = EqualityConfig(
        show_difference=True, reporter=lambda msg, path: reports.append((msg, path))
    )
    SetSameValuesHandler(next_handler=TrueHandler()).handle({1.5}, {2.5}, config)
    assert reports == [
        ("sets have different elements: 1.5 of actual has no match in expected", (1.5,))
    ]


def test_set_same_values_handler_handle_show_difference_different_lengths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = EqualityConfig(show_difference=True)
    with caplog.at_level(logging.INFO):
        SetSameValuesHandler(next_handler=TrueHandler()).handle({1, 2}, {1}, config)
    assert caplog.messages == ["sets have different lengths: 2 vs 1"]


def test_set_same_values_handler_handle_show_difference_does_not_log_when_equal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = EqualityConfig(show_difference=True)
    with caplog.at_level(logging.INFO):
        SetSameValuesHandler(next_handler=TrueHandler()).handle({1.5}, {1.5}, config)
    assert not caplog.messages


class BadRepr:
    r"""A hashable object whose ``repr`` raises."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BadRepr) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        msg = "broken __repr__"
        raise ZeroDivisionError(msg)


def test_set_same_values_handler_handle_element_with_failing_repr() -> None:
    reports: list[str] = []
    config = EqualityConfig(show_difference=True, reporter=lambda msg, _path: reports.append(msg))
    SetSameValuesHandler(next_handler=TrueHandler()).handle(
        frozenset({BadRepr(1)}), frozenset({BadRepr(2)}), config
    )
    assert reports[-1].startswith("sets have different elements: <BadRepr object at 0x")
