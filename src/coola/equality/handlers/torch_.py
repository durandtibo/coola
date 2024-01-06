r"""Implement some handlers for ``torch.Tensor``s."""

from __future__ import annotations

__all__ = ["TensorEqualHandler", "TensorSameDeviceHandler"]

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler
from coola.utils import is_torch_available

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class TensorEqualHandler(BaseEqualityHandler):
    r"""Check if the two tensors are equal.

    This handler returns ``True`` if the two tensors are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon
    >>> import torch
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers.torch_ import TensorEqualHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = TensorEqualHandler()
    >>> handler.handle(torch.ones(2, 3), torch.ones(2, 3), config)
    True
    >>> handler.handle(torch.ones(2, 3), torch.zeros(2, 3), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: torch.Tensor,
        object2: torch.Tensor,
        config: EqualityConfig,
    ) -> bool | None:
        if config.equal_nan:
            object_equal = object1.allclose(object2, atol=0.0, rtol=0.0, equal_nan=True)
        else:
            object_equal = object1.equal(object2)
        if config.show_difference and not object_equal:
            logger.info(
                f"torch.Tensors have different elements:\n"
                f"object1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class TensorSameDeviceHandler(AbstractEqualityHandler):
    r"""Check if the two tensors have the same device.

    This handler returns ``False`` if the two objects have different
    devices, otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon
    >>> import torch
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import TrueHandler
    >>> from coola.equality.handlers.torch_ import TensorSameDeviceHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = TensorSameDeviceHandler(next_handler=TrueHandler())
    >>> handler.handle(torch.ones(2, 3), torch.ones(3, 2), config)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: torch.Tensor,
        object2: torch.Tensor,
        config: EqualityConfig,
    ) -> bool | None:
        if object1.device != object2.device:
            if config.show_difference:
                logger.info(
                    f"torch.Tensors have different devices: {object1.device} vs {object2.device}"
                )
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)