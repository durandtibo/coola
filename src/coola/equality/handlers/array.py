r"""Implement some handlers for arrays or similar data."""

from __future__ import annotations

__all__ = ["ArraySameDTypeHandler"]

import logging
from typing import TYPE_CHECKING

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


class ArraySameDTypeHandler(AbstractEqualityHandler):
    r"""Check if the two arrays have the same data type.

    This handler returns ``False`` if the two objects have different
    data types, otherwise it passes the inputs to the next handler.
    This handler works on ``numpy.ndarray``s and ``torch.Tensor``s
    objects.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import ArraySameDTypeHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = ArraySameDTypeHandler()
    >>> handler.handle(np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: np.ndarray | torch.Tensor,
        object2: np.ndarray | torch.Tensor,
        config: EqualityConfig,
    ) -> bool | None:
        if object1.dtype != object2.dtype:
            if config.show_difference:
                logger.info(
                    f"objects have different data types: {object1.shape} vs {object2.shape}"
                )
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)
