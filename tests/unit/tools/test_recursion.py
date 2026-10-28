from __future__ import annotations

import pytest

from coola.tools.recursion import raise_recursion_limit_error

#################################################
#     Tests for raise_recursion_limit_error     #
#################################################


def test_raise_recursion_limit_error_raises_a_recursion_error() -> None:
    with pytest.raises(RecursionError):
        raise_recursion_limit_error("comparing", RecursionError())


def test_raise_recursion_limit_error_message() -> None:
    with pytest.raises(RecursionError) as exc_info:
        raise_recursion_limit_error("comparing", RecursionError())
    assert str(exc_info.value) == (
        "The Python recursion limit was exceeded while comparing deeply nested objects. "
        "Each nesting level uses several interpreter frames; reduce the nesting or increase "
        "the limit with sys.setrecursionlimit()."
    )


@pytest.mark.parametrize("action", ["hashing", "iterating over", "summarizing"])
def test_raise_recursion_limit_error_names_the_action(action: str) -> None:
    with pytest.raises(RecursionError, match=f"while {action} deeply nested"):
        raise_recursion_limit_error(action, RecursionError())


def test_raise_recursion_limit_error_sets_the_cause() -> None:
    cause = RecursionError("original")
    with pytest.raises(RecursionError) as exc_info:
        raise_recursion_limit_error("comparing", cause)
    assert exc_info.value.__cause__ is cause
