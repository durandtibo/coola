r"""Define some utility functions/classes."""

from __future__ import annotations

__all__ = ["ExamplePair", "RecursiveListTester", "legacy_registry", "messages", "run_tester"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from coola.equality.engine import Engine
from coola.equality.tester import BaseEqualityTester, get_default_registry

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig
    from coola.equality.tester.registry import EqualityTesterRegistry


@dataclass
class ExamplePair:
    actual: Any
    expected: Any
    expected_message: str | None = None
    atol: float = 0.0
    rtol: float = 0.0


class BadStr:
    r"""An object that is compared by value and whose ``__str__``
    raises."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BadStr) and self.value == other.value

    def __str__(self) -> str:
        msg = "broken __str__"
        raise ZeroDivisionError(msg)


def messages(config: EqualityConfig) -> list[str]:
    r"""Return the messages of the differences reported while ``config``
    was used, innermost first."""
    return [message for message, _ in config.reports or [] if message]


def run_tester(
    tester: BaseEqualityTester[Any], actual: Any, expected: Any, config: EqualityConfig
) -> bool:
    r"""Run a comparison where ``tester`` is the tester of the type of
    ``actual``.

    The engine checks the identity and the type before it calls the
    tester, and compares the child pairs of a container.
    """
    # Local imports: ``tests.unit.equality.utils`` is imported by the test modules at the top.
    from coola.equality.tester import get_default_registry

    registry = get_default_registry().copy()
    registry.register(type(actual), tester, exist_ok=True)
    return Engine(registry, config).run(actual, expected)


class RecursiveListTester(BaseEqualityTester[list]):
    r"""A tester that recurses through the registry, like the chain
    testers: it does not use the engine, so a deep nesting reaches the
    interpreter recursion limit."""

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(self, actual: list, expected: list, config: EqualityConfig, /) -> bool:
        return all(
            config.registry.objects_are_equal(a, e, config)
            for a, e in zip(actual, expected, strict=False)
        )


def legacy_registry() -> EqualityTesterRegistry:
    r"""Return a registry where the lists are compared with a tester
    that recurses."""
    registry = get_default_registry().copy()
    registry.register(list, RecursiveListTester(), exist_ok=True)
    return registry
