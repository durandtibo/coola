r"""Implement the engine that runs a comparison with an explicit stack.

The engine owns the invariants of a comparison: identity, strict type,
depth, cycles and the location of the first difference. A tester that
uses the engine never recurses: it returns a verdict, or the child pairs
that must all be equal.
"""

from __future__ import annotations

__all__ = ["Element", "Engine"]

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from coola.equality._helpers.format import format_type_difference
from coola.equality._helpers.utils import MaxDepthExceededError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from coola.equality.config import EqualityConfig
    from coola.equality.tester.base import BaseEqualityTester


@dataclass(frozen=True, slots=True)
class Element:
    r"""Location of an element of a set in the path of a difference.

    Args:
        value: The element.

    Example:
        ```pycon
        >>> from coola.equality import Element
        >>> Element(2)
        Element(value=2)

        ```
    """

    value: object


class Engine:
    r"""Run the comparison of two objects.

    One engine runs one comparison. For each pair of objects, the engine
    checks the identity, the strict type, finds the tester (the lookup
    is cached for the call) and calls it. A tester that uses the engine
    returns either a ``bool`` (the verdict) or an iterator of
    ``(actual, expected, path part)`` child pairs that must all be
    equal. The iterators are kept on an explicit stack, depth first, so
    the first difference in iteration order is the one found and
    nothing after it is visited. The stack use is constant: there is
    no ``RecursionError`` whatever the nesting.

    Args:
        config: The settings of the comparison (the registry that finds
            the tester of a type, the tolerance, ``equal_nan`` and
            ``max_depth``). A config is immutable and can be shared.
        explaining: If ``True``, the reasons are built and the first
            difference is recorded in ``failure``.

    Attributes:
        atol: The absolute tolerance.
        rtol: The relative tolerance.
        equal_nan: If ``True``, two NaNs are equal.
        explaining: If ``True``, the reasons are built.
        failure: ``None`` or, after a difference was found, the pair
            ``(path, reason)`` of the first difference.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> Engine(EqualityConfig()).run([1, {"a": 2}], [1, {"a": 2}])
        True
        >>> Engine(EqualityConfig()).run([1, {"a": 2}], [1, {"a": 3}])
        False

        ```
    """

    def __init__(self, config: EqualityConfig, *, explaining: bool = False) -> None:
        self.registry = config.registry
        self.config = config
        self.atol = config.atol
        self.rtol = config.rtol
        self.equal_nan = config.equal_nan
        self.max_depth = config.max_depth
        self.explaining = explaining
        self.failure: tuple[tuple[object, ...], str | None] | None = None
        self._testers: dict[type, BaseEqualityTester[Any]] = {}
        self._warned: set[Any] = set()
        # The path parts given by ``fail`` and its reason, read when the loop ends.
        self._extra: tuple[object, ...] = ()
        self._reason: str | Callable[[], str] | None = None

    def find(self, data_type: type) -> BaseEqualityTester[Any]:
        r"""Find the tester of a type.

        The registry lookup is paid once per type and per comparison.

        Args:
            data_type: The type of the objects.

        Returns:
            The tester of the type.
        """
        tester = self._testers.get(data_type)
        if tester is None:
            tester = self._testers[data_type] = self.registry.find_equality_tester(data_type)
        return tester

    def fail(self, reason: str | Callable[[], str], *path: object) -> bool:
        r"""Record the reason of the difference and return ``False``.

        A tester writes ``return ctx.fail(...)``.

        Args:
            reason: The reason, or a callable that builds it. A
                callable is called only if ``explaining`` is ``True``.
            *path: The path parts to add after the ones the engine
                knows, for example the index in an array.

        Returns:
            ``False``.

        Example:
            ```pycon
            >>> from coola.equality.config import EqualityConfig
            >>> from coola.equality.engine import Engine
            >>> from coola.equality.tester import get_default_registry
            >>> engine = Engine(EqualityConfig())
            >>> engine.fail("values differ", 3)
            False

            ```
        """
        self._extra = path
        self._reason = reason
        return False

    def warn_once(self, key: object, message: str) -> bool:
        r"""Emit a ``RuntimeWarning`` once per comparison.

        Args:
            key: The key that identifies the warning.
            message: The message of the warning.

        Returns:
            ``True`` if the warning was emitted, i.e. the first time ``key`` is given in the comparison,
                otherwise ``False``.
        """
        if key in self._warned:
            return False
        self._warned.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return True

    def run(self, actual: object, expected: object) -> bool:
        r"""Run the comparison.

        Args:
            actual: The actual object.
            expected: The expected object.

        Returns:
            ``True`` if the objects are equal, otherwise ``False``. In
                the second case, ``failure`` holds the path and the
                reason of the first difference.

        Raises:
            MaxDepthExceededError: If the nesting of the objects
                exceeds ``max_depth``.
        """
        testers = self._testers
        stack: list[Iterator[tuple[object, object, object]]] = []
        keys: list[tuple[int, int]] = []
        parts: list[object] = []
        visiting: set[tuple[int, int]] = set()
        pair: tuple[object, object, object] = (actual, expected, None)
        while True:
            actual, expected, part = pair
            # The loop is the hot path: the pair is tested inline.
            if actual is expected:
                result: bool | Iterator[tuple[object, object, object]] = True
            else:
                data_type = type(actual)
                tester: Any = testers.get(data_type) or self.find(data_type)
                if data_type is not type(expected):
                    result = self._fail_types(data_type, type(expected))
                else:
                    result = tester.objects_are_equal(actual, expected, self)
            if result is False:
                return self._record_failure(parts, part)
            if result is not True:
                # Child pairs. A pair that is already on the stack is equal (a cycle).
                key = (id(actual), id(expected))
                if key not in visiting:
                    self._check_depth(len(stack))
                    visiting.add(key)
                    stack.append(iter(result))
                    keys.append(key)
                    parts.append(part)
            while stack:  # the next pair: the top iterator, popping the exhausted ones
                nxt = next(stack[-1], None)
                if nxt is not None:
                    pair = nxt
                    break
                stack.pop()
                visiting.discard(keys.pop())
                parts.pop()
            else:
                return True

    def _fail_types(self, actual_type: type, expected_type: type) -> bool:
        r"""Record that the two objects have different types.

        Args:
            actual_type: The type of the actual object.
            expected_type: The type of the expected object.

        Returns:
            Always ``False``.
        """
        return self.fail(lambda: format_type_difference(actual_type, expected_type))

    def _check_depth(self, depth: int) -> None:
        r"""Raise an error if the nesting of the objects is too deep.

        Args:
            depth: The current nesting depth.

        Raises:
            MaxDepthExceededError: If ``depth`` reaches ``max_depth``.
        """
        if depth >= self.max_depth:
            msg = (
                f"Maximum recursion depth ({self.max_depth}) exceeded. "
                "Consider increasing max_depth parameter or simplifying the data structure."
            )
            raise MaxDepthExceededError(msg)

    def _record_failure(self, parts: list[object], part: object) -> bool:
        r"""Record the first difference in ``failure``.

        A reason given to ``fail`` is recorded with the path of the
        stack and the parts given to ``fail``.

        Args:
            parts: The path parts of the stack of the engine.
            part: The path part of the failing pair.

        Returns:
            Always ``False``.
        """
        if not self.explaining:
            return False
        # ``parts[0]`` is the part of the root, which is not in the path. The failing pair is
        # the root if the stack is empty.
        path = (*parts[1:], part) if parts else ()
        reason = (
            self._reason
            if isinstance(self._reason, str) or self._reason is None
            else self._reason()
        )
        self.failure = (path + self._extra, reason)
        return False
