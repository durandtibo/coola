r"""Implement some handlers for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["NumpyArrayEqualHandler"]

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.equality.handlers.base import BaseEqualityHandler
from coola.utils import is_numpy_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class NumpyArrayEqualHandler(BaseEqualityHandler):
    r"""Check if the two NumPy arrays are equal.

    This handler returns ``True`` if the two arrays are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon
    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import NumpyArrayEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = NumpyArrayEqualHandler()
    >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
    True
    >>> handler.handle(np.ones((2, 3)), np.zeros((2, 3)), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: np.ndarray,
        object2: np.ndarray,
        config: EqualityConfig,
    ) -> bool:
        object_equal = np.array_equal(object1, object2, equal_nan=config.equal_nan)
        if config.show_difference and not object_equal:
            logger.info(
                f"numpy.ndarrays have different elements:\n"
                f"object1:\n{object1}\nobject2:\n{object2}"
            )
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.
