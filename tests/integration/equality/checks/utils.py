from __future__ import annotations

from coola.equality import objects_are_allclose, objects_are_equal
from coola.equality.config import EqualityConfig
from coola.equality.interface import run_comparison
from coola.equality.tester import get_default_registry

EQUALITY_TESTER_FUNCTIONS = [objects_are_equal, objects_are_allclose]


def reported_messages(actual: object, expected: object) -> list[str]:
    r"""Return the messages the testers report when the objects are
    different, innermost first."""
    config = EqualityConfig(reports=[])
    run_comparison(get_default_registry(), actual, expected, config)
    return [message for message, _ in config.reports or [] if message]
