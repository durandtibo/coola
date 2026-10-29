r"""Define some utility functions/classes."""

from __future__ import annotations

__all__ = [
    "ExamplePair",
    "ExplainingConfig",
    "explaining",
    "messages",
    "run_tester",
]

from dataclasses import dataclass, field, replace
from typing import Any

from coola.equality.config import EqualityConfig
from coola.equality.engine import Engine
from coola.equality.tester import BaseEqualityTester, get_default_registry


@dataclass
class ExamplePair:
    actual: Any
    expected: Any
    expected_message: str = ""
    atol: float = 0.0
    rtol: float = 0.0


class BadStr:
    r"""An object that is compared by value and whose ``__str__``
    raises."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BadStr) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        msg = "broken __str__"
        raise ZeroDivisionError(msg)


@dataclass(frozen=True, slots=True)
class ExplainingConfig(EqualityConfig):
    r"""A config for the tests: ``run_tester`` explains the difference
    and records the ``(reason, path)`` in ``reports``."""

    reports: list[tuple[str, tuple[object, ...]]] = field(default_factory=list, compare=False)


def explaining(config: EqualityConfig) -> ExplainingConfig:
    r"""Return a config like ``config`` that records the explanation of
    the differences."""
    return ExplainingConfig(
        registry=config.registry,
        tolerance=config.tolerance,
        equal_nan=config.equal_nan,
        max_depth=config.max_depth,
    )


def messages(config: EqualityConfig) -> list[str]:
    r"""Return the reasons of the differences recorded while ``config``
    was used."""
    return [message for message, _ in getattr(config, "reports", []) if message]


def run_tester(
    tester: BaseEqualityTester[Any], actual: Any, expected: Any, config: EqualityConfig
) -> bool:
    r"""Run a comparison where ``tester`` is the tester of the type of
    ``actual``.

    The engine checks the identity and the type before it calls the
    tester, and compares the child pairs of a container.
    """
    # Local imports: ``tests.unit.equality.utils`` is imported by the test modules at the top.

    registry = get_default_registry().copy()
    registry.register(type(actual), tester, exist_ok=True)
    engine = Engine(
        replace(config, registry=registry), explaining=isinstance(config, ExplainingConfig)
    )
    equal = engine.run(actual, expected)
    if isinstance(config, ExplainingConfig) and engine.failure is not None:
        path, reason = engine.failure
        config.reports.append((reason or "", path))
    return equal
