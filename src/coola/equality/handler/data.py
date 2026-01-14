r"""Implement handler to check the objects have the same data."""

from __future__ import annotations

__all__ = ["SameDataHandler", "SupportsData"]

import logging
from typing import TYPE_CHECKING, Any, Protocol

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.utils import handlers_are_equal

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger: logging.Logger = logging.getLogger(__name__)


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


class SameDataHandler(BaseEqualityHandler):
    r"""Check if the two objects have the same data.

    This handler returns ``False`` if the two objects have different
    data, otherwise it passes the inputs to the next handler.
    The objects must have a ``data`` attribute (e.g. ``object.data``)
    which returns the data of the object.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import SameDataHandler, TrueHandler
        >>> config = EqualityConfig()
        >>> handler = SameDataHandler(next_handler=TrueHandler())
        >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
        True
        >>> handler.handle(np.ones((2, 3)), np.zeros((2, 3)), config)
        False

        ```
    """

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return handlers_are_equal(self.next_handler, other.next_handler)

    def handle(self, actual: SupportsData, expected: SupportsData, config: EqualityConfig) -> bool:
        if not config.registry.objects_are_equal(actual.data, expected.data, config):
            if config.show_difference:
                logger.info(f"objects have different data: {actual.data} vs {expected.data}")
            return False
        return self._handle_next(actual, expected, config=config)
