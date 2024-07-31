r"""Implement handlers to check the objects have the same data."""

from __future__ import annotations

__all__ = ["SameDataHandler", "SupportsData"]

import logging
from typing import TYPE_CHECKING, Any, Protocol

from coola.equality.handlers.base import AbstractEqualityHandler

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger = logging.getLogger(__name__)


class SupportsData(Protocol):
    r"""Implement a protocol to represent objects with a ``data``
    attribute.

    This protocol can be used to represent several objects like
    ``jax.numpy.ndarray``s, ``numpy.ndarray``s,  and
    ``torch.Tensor``s.
    """

    @property
    def data(self) -> Any:
        return  # pragma: no cover


class SameDataHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same data.

    This handler returns ``False`` if the two objects have different
    data, otherwise it passes the inputs to the next handler.
    The objects must have a ``data`` attribute (e.g. ``object.data``)
    which returns the shape of the object.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameDataHandler, TrueHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameDataHandler(next_handler=TrueHandler())
    >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
    True
    >>> handler.handle(np.ones((2, 3)), np.zeros((2, 3)), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(self, actual: SupportsData, expected: SupportsData, config: EqualityConfig) -> bool:
        if not config.tester.equal(actual.data, expected.data, config):
            if config.show_difference:
                logger.info(f"objects have different data: {actual.data} vs {expected.data}")
            return False
        return self._handle_next(actual, expected, config=config)
