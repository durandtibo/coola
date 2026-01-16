r"""Implement handlers for ``torch.Tensor``s."""

from __future__ import annotations

__all__ = ["TorchTensorEqualHandler", "TorchTensorSameDeviceHandler"]

import logging
from typing import TYPE_CHECKING

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.utils import handlers_are_equal

if TYPE_CHECKING:
    import torch

    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class TorchTensorEqualHandler(BaseEqualityHandler):
    r"""Check if the two tensors are equal.

    This handler returns ``True`` if the two tensors are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import TorchTensorEqualHandler
        >>> config = EqualityConfig()
        >>> handler = TorchTensorEqualHandler()
        >>> handler.handle(torch.ones(2, 3), torch.ones(2, 3), config)
        True
        >>> handler.handle(torch.ones(2, 3), torch.zeros(2, 3), config)
        False

        ```
    """

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return handlers_are_equal(self.next_handler, other.next_handler)

    def handle(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        config: EqualityConfig,
    ) -> bool:
        # Import torch here to handle optional dependency
        import torch  # noqa: PLC0415

        # Validate that both inputs are actually torch.Tensor
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            msg = (
                f"Expected both inputs to be torch.Tensor, but got "
                f"{type(actual).__name__} and {type(expected).__name__}"
            )
            raise TypeError(msg)

        object_equal = tensor_equal(actual, expected, config)
        if config.show_difference and not object_equal:
            # Avoid printing very large tensors
            if actual.numel() > 10000 or expected.numel() > 10000:
                logger.info(
                    f"torch.Tensors have different elements (tensors too large to display, "
                    f"shapes: {actual.shape} vs {expected.shape})"
                )
            else:
                logger.info(
                    f"torch.Tensors have different elements:\n"
                    f"actual=\n{actual}\nexpected=\n{expected}"
                )
        return object_equal


class TorchTensorSameDeviceHandler(BaseEqualityHandler):
    r"""Check if the two tensors have the same device.

    This handler returns ``False`` if the two objects have different
    devices, otherwise it passes the inputs to the next handler.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import TrueHandler, TorchTensorSameDeviceHandler
        >>> config = EqualityConfig()
        >>> handler = TorchTensorSameDeviceHandler(next_handler=TrueHandler())
        >>> handler.handle(torch.ones(2, 3), torch.ones(3, 2), config)
        True

        ```
    """

    def equal(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return handlers_are_equal(self.next_handler, other.next_handler)

    def handle(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        config: EqualityConfig,
    ) -> bool:
        # Import torch here to handle optional dependency
        import torch  # noqa: PLC0415

        # Validate that both inputs are actually torch.Tensor
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            msg = (
                f"Expected both inputs to be torch.Tensor, but got "
                f"{type(actual).__name__} and {type(expected).__name__}"
            )
            raise TypeError(msg)

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
        ``True`` if the two tensors are equal within a tolerance,
            otherwise ``False``.
    """
    if config.equal_nan or config.atol > 0 or config.rtol > 0:
        return tensor1.allclose(
            tensor2, atol=config.atol, rtol=config.rtol, equal_nan=config.equal_nan
        )
    return tensor1.equal(tensor2)
