r"""Implement some handlers for ``torch.Tensor``s."""

from __future__ import annotations

__all__ = ["TorchTensorEqualHandler", "TorchTensorSameDeviceHandler"]

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


class TorchTensorEqualHandler(BaseEqualityHandler):
    r"""Check if the two tensors are equal.

    This handler returns ``True`` if the two tensors are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon

    >>> import torch
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers.torch_ import TorchTensorEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = TorchTensorEqualHandler()
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
        actual: torch.Tensor,
        expected: torch.Tensor,
        config: EqualityConfig,
    ) -> bool:
        object_equal = tensor_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            logger.info(
                f"torch.Tensors have different elements:\n"
                f"actual=\n{actual}\nexpected=\n{expected}"
            )
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class TorchTensorSameDeviceHandler(AbstractEqualityHandler):
    r"""Check if the two tensors have the same device.

    This handler returns ``False`` if the two objects have different
    devices, otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon

    >>> import torch
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import TrueHandler
    >>> from coola.equality.handlers.torch_ import TorchTensorSameDeviceHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = TorchTensorSameDeviceHandler(next_handler=TrueHandler())
    >>> handler.handle(torch.ones(2, 3), torch.ones(3, 2), config)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        config: EqualityConfig,
    ) -> bool | None:
        if actual.device != expected.device:
            if config.show_difference:
                logger.info(
                    f"torch.Tensors have different devices: {actual.device} vs {expected.device}"
                )
            return False
        return self._handle_next(actual, expected, config=config)


def tensor_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, config: EqualityConfig) -> bool:
    r"""Indicate if the two tensors are equal within a tolerance.

    Args:
        tensor1: The first tensor to compare.
        tensor2: The second tensor to compare.
        config: The equality configuration.

    Returns:
        ``True``if the two tensors are equal within a tolerance,
            otherwise ``False``.
    """
    if config.equal_nan or config.atol > 0 or config.rtol > 0:
        return tensor1.allclose(
            tensor2, atol=config.atol, rtol=config.rtol, equal_nan=config.equal_nan
        )
    return tensor1.equal(tensor2)
