r"""Implement handler to check the objects have the same data type."""

from __future__ import annotations

__all__ = ["SameDTypeHandler", "SupportsDType"]

import logging
from typing import TYPE_CHECKING, Any, Protocol

from coola.equality.handler.base import AbstractEqualityHandler

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig2


logger: logging.Logger = logging.getLogger(__name__)


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


class SameDTypeHandler(AbstractEqualityHandler):  # noqa: PLW1641
    r"""Check if the two objects have the same data type.

    This handler returns ``False`` if the two objects have different
    data types, otherwise it passes the inputs to the next handler.
    The objects must have a ``dtype`` attribute (e.g. ``object.dtype``)
    which returns the data type of the object. This handler works on
    ``numpy.ndarray``s and ``torch.Tensor``s objects.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.handler import SameDTypeHandler, TrueHandler
        >>> config = EqualityConfig2()
        >>> handler = SameDTypeHandler(next_handler=TrueHandler())
        >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
        True
        >>> handler.handle(np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int), config)
        False

        ```
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def handle(
        self, actual: SupportsDType, expected: SupportsDType, config: EqualityConfig2
    ) -> bool:
        if actual.dtype != expected.dtype:
            if config.show_difference:
                logger.info(
                    f"objects have different data types: {actual.dtype} vs {expected.dtype}"
                )
            return False
        return self._handle_next(actual, expected, config=config)
