r"""Implement a handler to check if two objects have the same shape."""

from __future__ import annotations

__all__ = ["SameShapeHandler", "SupportsShape"]

import logging
from typing import TYPE_CHECKING, Protocol

from coola.equality.handlers.base import AbstractEqualityHandler

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger = logging.getLogger(__name__)


class SupportsShape(Protocol):
    r"""Implement a protocol to represent objects with a ``shape``
    attribute.

    This protocol can be used to represent several objects like
    ``jax.numpy.ndarray``s, ``numpy.ndarray``s, ``pandas.DataFrame``,
    ``polars.DataFrame`` and ``torch.Tensor``s.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return ()  # pragma: no cover


class SameShapeHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same shape.

    This handler returns ``False`` if the two objects have different
    shapes, otherwise it passes the inputs to the next handler.
    The objects must have a ``shape`` attribute (e.g. ``object.shape``)
    which returns the shape of the object. This handler works on
    ``jax.numpy.ndarray``s, ``numpy.ndarray``s, ``pandas.DataFrame``,
    ``polars.DataFrame`` and ``torch.Tensor``s objects.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameShapeHandler, TrueHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameShapeHandler(next_handler=TrueHandler())
    >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
    True
    >>> handler.handle(np.ones((2, 3)), np.ones((3, 2)), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(
        self, actual: SupportsShape, expected: SupportsShape, config: EqualityConfig
    ) -> bool:
        if actual.shape != expected.shape:
            if config.show_difference:
                logger.info(f"objects have different shapes: {actual.shape} vs {expected.shape}")
            return False
        return self._handle_next(actual, expected, config=config)
