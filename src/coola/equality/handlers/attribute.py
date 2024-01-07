r"""Implement a handler to check if two objects have the same
attribute."""

from __future__ import annotations

__all__ = ["SameAttributeHandler"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler
from coola.utils import repr_indent, repr_mapping

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger = logging.getLogger(__name__)


class SameAttributeHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same attribute.

    This handler returns ``False`` if the two objects have different
    attributes, otherwise it passes the inputs to the next handler.
    The objects must have the attribute.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameAttributeHandler, TrueHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"name": self._name, "next_handler": self._next_handler}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(name={self._name})"

    @property
    def name(self) -> str:
        return self._name

    def handle(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
        value1 = getattr(object1, self._name)
        value2 = getattr(object2, self._name)
        if not config.tester.equal(value1, value2, config.show_difference):
            if config.show_difference:
                logger.info(f"objects have different {self._name}: {value1} vs {value2}")
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)
