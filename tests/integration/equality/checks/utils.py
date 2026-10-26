from __future__ import annotations

from coola.equality import compare, objects_are_allclose, objects_are_equal

EQUALITY_TESTER_FUNCTIONS = [objects_are_equal, objects_are_allclose]


def reported_messages(actual: object, expected: object) -> list[str]:
    r"""Return the reason of the first difference as a list (empty if the
    objects are equal)."""
    reason = compare(actual, expected).reason
    return [reason] if reason else []
