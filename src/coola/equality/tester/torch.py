r"""Implement an equality tester for ``torch.Tensor``s and
``torch.nn.utils.rnn.PackedSequence``s."""

from __future__ import annotations

__all__ = ["TorchPackedSequenceEqualityTester", "TorchTensorEqualityTester"]

import logging
from typing import TYPE_CHECKING

from coola.equality.handler import (
    SameAttributeHandler,
    SameDataHandler,
    SameDTypeHandler,
    SameObjectHandler,
    SameShapeHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.equality.handler.torch import (
    TorchTensorEqualHandler,
    TorchTensorSameDeviceHandler,
)
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_torch, is_torch_available

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class TorchPackedSequenceEqualityTester(BaseEqualityTester[torch.nn.utils.rnn.PackedSequence]):
    r"""Implement an equality tester for ``torch.Tensor``.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import TorchPackedSequenceEqualityTester
        >>> config = EqualityConfig()
        >>> tester = TorchTensorEqualityTester()
        >>> tester.objects_are_equal(torch.ones(2, 3), torch.ones(2, 3), config)
        True
        >>> tester.objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3), config)
        False

        ```
    """

    def __init__(self) -> None:
        check_torch()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameDataHandler()).chain(
            SameAttributeHandler(name="batch_sizes")
        ).chain(SameAttributeHandler(name="sorted_indices")).chain(
            SameAttributeHandler(name="unsorted_indices")
        ).chain(TrueHandler())  # fmt: skip

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class TorchTensorEqualityTester(BaseEqualityTester[torch.Tensor]):
    r"""Implement an equality tester for ``torch.Tensor``.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import TorchTensorEqualityTester
        >>> config = EqualityConfig()
        >>> tester = TorchTensorEqualityTester()
        >>> tester.objects_are_equal(torch.ones(2, 3), torch.ones(2, 3), config)
        True
        >>> tester.objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3), config)
        False

        ```
    """

    def __init__(self) -> None:
        check_torch()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameDTypeHandler()).chain(
            SameShapeHandler()
        ).chain(TorchTensorSameDeviceHandler()).chain(TorchTensorEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
