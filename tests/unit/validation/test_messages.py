from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.validation import (
    validate_finite,
    validate_ge,
    validate_gt,
    validate_in,
    validate_int,
    validate_isinstance,
    validate_le,
    validate_lt,
    validate_not_empty,
    validate_number,
    validate_range,
)

if TYPE_CHECKING:
    from collections.abc import Callable

NOT_A_NUMBER: Any = "a"
NOT_SIZED: Any = 5

# One failing call per validator: the call, the exception type and the text that the message
# must contain (the received value, or the name of the received type).
FAILING_CALLS: list[tuple[Callable[[], None], type[Exception], str]] = [
    (lambda: validate_isinstance("a", int, name="x"), TypeError, "got str"),
    (lambda: validate_int(1.5, name="x"), TypeError, "got float"),
    (lambda: validate_number("a", name="x"), TypeError, "got str"),
    (lambda: validate_ge(-1, 0, name="x"), ValueError, "got -1"),
    (lambda: validate_gt(-1, 0, name="x"), ValueError, "got -1"),
    (lambda: validate_le(2, 0, name="x"), ValueError, "got 2"),
    (lambda: validate_lt(2, 0, name="x"), ValueError, "got 2"),
    (lambda: validate_range(2, 0, 1, name="x"), ValueError, "got 2"),
    (lambda: validate_finite(float("inf"), name="x"), ValueError, "got inf"),
    (lambda: validate_finite(NOT_A_NUMBER, name="x"), TypeError, "got str"),
    (lambda: validate_not_empty([], name="x"), ValueError, "x must not be empty"),
    (lambda: validate_not_empty(NOT_SIZED, name="x"), TypeError, "got int"),
    (lambda: validate_in(5, (1, 2), name="x"), ValueError, "got 5"),
    (lambda: validate_ge(None, 0, name="x"), TypeError, "got None"),
]


CALLS_AND_ERRORS = [(call, error) for call, error, _ in FAILING_CALLS]


@pytest.mark.parametrize(("call", "error"), CALLS_AND_ERRORS)
def test_message_starts_with_the_name(call: Callable[[], None], error: type[Exception]) -> None:
    with pytest.raises(error, match=r"^x (must|cannot)"):
        call()


@pytest.mark.parametrize(("call", "error", "received"), FAILING_CALLS)
def test_message_contains_the_value_received(
    call: Callable[[], None], error: type[Exception], received: str
) -> None:
    with pytest.raises(error, match=received):
        call()


@pytest.mark.parametrize(("call", "error"), CALLS_AND_ERRORS)
def test_message_is_bounded(call: Callable[[], None], error: type[Exception]) -> None:
    with pytest.raises(error) as exc_info:
        call()
    assert len(str(exc_info.value)) < 200


@pytest.mark.parametrize(
    "call",
    [
        lambda: validate_isinstance(1, int),
        lambda: validate_int(1),
        lambda: validate_number(1.5),
        lambda: validate_ge(1, 0),
        lambda: validate_gt(1, 0),
        lambda: validate_le(0, 1),
        lambda: validate_lt(0, 1),
        lambda: validate_range(0.5, 0, 1),
        lambda: validate_finite(1.5),
        lambda: validate_not_empty([1]),
        lambda: validate_in(1, (1, 2)),
    ],
)
def test_validator_is_silent_on_success(call: Callable[[], Any]) -> None:
    assert call() is None
