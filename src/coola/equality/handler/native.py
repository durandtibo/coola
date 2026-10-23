r"""Implement handlers for native Python objects."""

from __future__ import annotations

__all__ = [
    "FalseHandler",
    "ObjectEqualHandler",
    "SameAttributeHandler",
    "SameLengthHandler",
    "SameObjectHandler",
    "SameTypeHandler",
    "TrueHandler",
]

import logging
from operator import attrgetter
from typing import TYPE_CHECKING

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import (
    format_type_difference,
    format_value_difference,
)
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import AttributeName, report_difference

if TYPE_CHECKING:
    from collections.abc import Sized

    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class FalseHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Implement a handler that always returns ``False``.

    This handler is designed to be used at the end of the chain of
    responsibility. This handler does not call the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import FalseHandler
        >>> config = EqualityConfig()
        >>> handler = FalseHandler()
        >>> handler.handle("abc", "abc", config)
        False
        >>> handler.handle("abc", "ABC", config)
        False

        ```
    """

    def handle(
        self,
        actual: object,  # noqa: ARG002
        expected: object,  # noqa: ARG002
        config: EqualityConfig,  # noqa: ARG002
    ) -> bool:
        return False


class TrueHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Implement a handler that always returns ``True``.

    This handler is designed to be used at the end of the chain of
    responsibility. This handler does not call the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import TrueHandler
        >>> config = EqualityConfig()
        >>> handler = TrueHandler()
        >>> handler.handle("abc", "abc", config)
        True
        >>> handler.handle("abc", "ABC", config)
        True

        ```
    """

    def handle(
        self,
        actual: object,  # noqa: ARG002
        expected: object,  # noqa: ARG002
        config: EqualityConfig,  # noqa: ARG002
    ) -> bool:
        return True


class ObjectEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects are equal using the default equality
    operator ``==``.

    This handler returns ``True`` if the two objects are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import ObjectEqualHandler
        >>> config = EqualityConfig()
        >>> handler = ObjectEqualHandler()
        >>> handler.handle(1, 1, config)
        True
        >>> handler.handle(1, "abc", config)
        False

        ```
    """

    def handle(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        try:
            result = actual == expected
            object_equal = bool(result)
        except (TypeError, ValueError) as exc:
            # e.g. a dataclass holding an array: ``==`` returns an array, or ``==`` itself
            # raises (a pydantic model holding an array). A ``TypeError`` is raised
            # instead of returning ``False``, because ``False`` would claim that the
            # objects are different when they were not compared at all.
            msg = (
                f"`==` raised or returned a value that cannot be converted to a bool "
                f"({exc}) when comparing {type(actual)} objects. Register an equality "
                "tester for this type."
            )
            raise TypeError(msg) from exc
        if config.show_difference and not object_equal:
            report_difference(
                logger, config, format_value_difference(actual=actual, expected=expected)
            )
        return object_equal


class SameAttributeHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects have the same attribute.

    This handler returns ``False`` if the two objects have different
    attributes, otherwise it passes the inputs to the next handler.
    The objects must have the attribute. The name can be a dotted
    path (for example ``"coords.variables"``) to compare the
    attribute of an attribute.

    Args:
        name: The name of the attribute to compare. It can be a dotted
            path (for example ``"coords.variables"``).
        next_handler: The next handler in the chain. Defaults to
            ``None``.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import SameAttributeHandler, TrueHandler
        >>> config = EqualityConfig()
        >>> handler = SameAttributeHandler(name="shape", next_handler=TrueHandler())
        >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
        True
        >>> handler.handle(np.ones((2, 3)), np.ones((3, 2)), config)
        False

        ```
    """

    def __init__(self, name: str, next_handler: BaseEqualityHandler | None = None) -> None:
        super().__init__(next_handler=next_handler)
        self._name = name

    def __repr__(self) -> str:
        args = ""
        if self.next_handler is not None:
            args += f", next_handler={self.next_handler!r}"
        return f"{self.__class__.__qualname__}(name={self._name}{args})"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(name={self._name})"

    @property
    def name(self) -> str:
        r"""The name of the attribute to compare."""
        return self._name

    def equality_attrs(self) -> tuple[str, ...]:
        return ("name",)

    def handle(self, actual: object, expected: object, config: EqualityConfig) -> bool:
        # ``attrgetter`` instead of ``getattr`` to support a dotted name
        # (e.g. ``"coords.variables"``): ``getattr(obj, "a.b")`` looks for an
        # attribute whose name contains a dot, while ``attrgetter("a.b")``
        # reads ``obj.a.b``. For a name without a dot, it is the same as ``getattr``.
        value1 = attrgetter(self._name)(actual)
        value2 = attrgetter(self._name)(expected)
        if not config.registry.objects_are_equal(value1, value2, config):
            if config.show_difference:
                report_difference(
                    logger,
                    config,
                    "objects have different %s: %s vs %s",
                    self._name,
                    value1,
                    value2,
                    extra={
                        "coola_path": tuple(AttributeName(name) for name in self._name.split(".")),
                    },
                )
            return False
        return self._handle_next(actual, expected, config=config)


class SameLengthHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects have the same length.

    This handler returns ``False`` if the two objects have different
    lengths, otherwise it passes the inputs to the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import SameLengthHandler
        >>> config = EqualityConfig()
        >>> handler = SameLengthHandler()
        >>> handler.handle([1, 2, 3], [1, 2, 3, 4], config)
        False

        ```
    """

    def handle(
        self,
        actual: Sized,
        expected: Sized,
        config: EqualityConfig,
    ) -> bool:
        if len(actual) != len(expected):
            if config.show_difference:
                report_difference(
                    logger,
                    config,
                    "objects have different lengths: %s vs %s",
                    f"{len(actual):,}",
                    f"{len(expected):,}",
                )
            return False
        return self._handle_next(actual, expected, config=config)


class SameObjectHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects refer to the same object.

    This handler returns ``True`` if the two objects refer to the
    same object, otherwise it passes the inputs to the next handler.

    Notes:
        The identity shortcut also applies to objects that are not equal
        to themselves, like ``float("nan")`` or a container holding it,
        even with ``equal_nan=False``.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import SameObjectHandler
        >>> config = EqualityConfig()
        >>> handler = SameObjectHandler()
        >>> handler.handle("abc", "abc", config)
        True

        ```
    """

    def handle(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        if actual is expected:
            return True
        return self._handle_next(actual, expected, config=config)


class SameTypeHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects have the same type.

    This handler returns ``False`` if the two objects have different
    types, otherwise it passes the inputs to the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import SameTypeHandler
        >>> config = EqualityConfig()
        >>> handler = SameTypeHandler()
        >>> handler.handle(1, "abc", config)
        False

        ```
    """

    def handle(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        if type(actual) is not type(expected):
            if config.show_difference:
                message = format_type_difference(type(actual), type(expected))
                report_difference(logger, config, message)
            return False
        return self._handle_next(actual, expected, config=config)
