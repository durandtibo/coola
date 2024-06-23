r"""Implement handlers to check the objects have the same data type."""

from __future__ import annotations

__all__ = ["SameDTypeHandler", "SupportsDType"]

import logging
from typing import TYPE_CHECKING, Any, Protocol

from coola.equality.handlers.base import AbstractEqualityHandler

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger = logging.getLogger(__name__)


class SupportsDType(Protocol):
    r"""Implement a protocol to represent objects with a ``dtype``
    attribute.

    This protocol can be used to represent several objects like
    ``jax.numpy.ndarray``s, ``numpy.ndarray``s,  and
    ``torch.Tensor``s.
    """

    @property
    def dtype(self) -> Any:
        return  # pragma: no cover


class SameDTypeHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same data type.

    This handler returns ``False`` if the two objects have different
    data types, otherwise it passes the inputs to the next handler.
    The objects must have a ``dtype`` attribute (e.g. ``object.dtype``)
    which returns the shape of the object. This handler works on
    ``numpy.ndarray``s and ``torch.Tensor``s objects.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameDTypeHandler, TrueHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameDTypeHandler(next_handler=TrueHandler())
    >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
    True
    >>> handler.handle(np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(
        self, actual: SupportsDType, expected: SupportsDType, config: EqualityConfig
    ) -> bool:
        if actual.dtype != expected.dtype:
            if config.show_difference:
                logger.info(
                    f"objects have different data types: {actual.dtype} vs {expected.dtype}"
                )
            return False
        return self._handle_next(actual, expected, config=config)
