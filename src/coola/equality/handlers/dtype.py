r"""Implement some handlers for arrays or similar data."""

from __future__ import annotations

__all__ = ["SameDTypeHandler"]

import logging
from typing import TYPE_CHECKING, Any, Protocol

from coola.equality.handlers.base import AbstractEqualityHandler

if TYPE_CHECKING:
    from unittest.mock import Mock

    from coola.equality.config import EqualityConfig
    from coola.utils import is_numpy_available, is_torch_available

    if is_numpy_available():
        import numpy as np
    else:  # pragma: no cover
        np = Mock()

    if is_torch_available():
        import torch
    else:  # pragma: no cover
        torch = Mock()

logger = logging.getLogger(__name__)


class SupportsDType(Protocol):
    r"""Implement a protocol to represent objects with a ``dtype``
    property.

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
    The objects must have a ``dtype`` property (e.g. ``object.dtype``)
    which returns the shape of the object. This handler works on
    ``numpy.ndarray``s and ``torch.Tensor``s objects.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameDTypeHandler, TrueHandler
    >>> from coola.testers import EqualityTester
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

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self, object1: SupportsDType, object2: SupportsDType, config: EqualityConfig
    ) -> bool:
        if object1.dtype != object2.dtype:
            if config.show_difference:
                logger.info(
                    f"objects have different data types: {object1.dtype} vs {object2.dtype}"
                )
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)
