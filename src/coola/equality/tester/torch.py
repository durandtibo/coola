r"""Implement equality testers for PyTorch tensors and packed sequences.

This module provides equality testers for torch.Tensor and
torch.nn.utils.rnn.PackedSequence with support for device, dtype, shape,
and value comparisons including NaN and tolerance handling.
"""

from __future__ import annotations

__all__ = ["TorchPackedSequenceEqualityTester", "TorchTensorEqualityTester"]

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


class TorchPackedSequenceEqualityTester(BaseEqualityTester[torch.nn.utils.rnn.PackedSequence]):
    r"""Implement an equality tester for ``torch.nn.utils.rnn.PackedSequence``.

    This tester compares PyTorch packed sequences by checking data tensor and
    metadata attributes. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are PackedSequence
    3. SameDataHandler: Compare the data tensors
    4. SameAttributeHandler("batch_sizes"): Compare batch_sizes tensors
    5. SameAttributeHandler("sorted_indices"): Compare sorted_indices (if present)
    6. SameAttributeHandler("unsorted_indices"): Compare unsorted_indices (if present)
    7. TrueHandler: Return True if all checks pass

    Example:
        Basic packed sequence comparison:

        ```pycon
        >>> import torch
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import TorchPackedSequenceEqualityTester
        >>> config = EqualityConfig()
        >>> tester = TorchPackedSequenceEqualityTester()
        >>> seq1 = torch.nn.utils.rnn.pack_padded_sequence(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     lengths=torch.tensor([2, 1]),
        ...     batch_first=True,
        ...     enforce_sorted=False,
        ... )
        >>> seq2 = torch.nn.utils.rnn.pack_padded_sequence(
        ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ...     lengths=torch.tensor([2, 1]),
        ...     batch_first=True,
        ...     enforce_sorted=False,
        ... )
        >>> tester.objects_are_equal(seq1, seq2, config)
        True

        ```
    """

    def __init__(self) -> None:
        """Initialize the PyTorch packed sequence equality tester.

        The handler chain performs checks in this order:
        1. SameObjectHandler: Quick check for object identity
        2. SameTypeHandler: Verify both are PackedSequence objects
        3. SameDataHandler: Compare the underlying data tensors
        4. SameAttributeHandler("batch_sizes"): Compare batch size tensors
        5. SameAttributeHandler("sorted_indices"): Compare sorted indices
        6. SameAttributeHandler("unsorted_indices"): Compare unsorted indices
        7. TrueHandler: Return True if all previous checks passed

        Raises:
            RuntimeError: If PyTorch is not installed.
        """
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
        actual: torch.nn.utils.rnn.PackedSequence,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class TorchTensorEqualityTester(BaseEqualityTester[torch.Tensor]):
    r"""Implement an equality tester for ``torch.Tensor``.

    This tester compares PyTorch tensors element-wise with support for device,
    dtype, shape checking, NaN equality, and tolerance-based comparisons. The
    handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify both are torch tensors
    3. SameDTypeHandler: Check tensors have the same dtype
    4. SameShapeHandler: Verify tensors have the same shape
    5. TorchTensorSameDeviceHandler: Ensure tensors are on the same device
    6. TorchTensorEqualHandler: Element-wise comparison with tolerance support

    The tester respects config.equal_nan for NaN comparisons and config.atol/rtol
    for floating-point tolerance.

    Example:
        Basic tensor comparison:

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

        NaN comparison with equal_nan:

        ```pycon
        >>> import torch
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import TorchTensorEqualityTester
        >>> config = EqualityConfig(equal_nan=True)
        >>> tester = TorchTensorEqualityTester()
        >>> tester.objects_are_equal(
        ...     torch.tensor([1.0, float("nan")]),
        ...     torch.tensor([1.0, float("nan")]),
        ...     config,
        ... )
        True

        ```

        Tensors on different devices are not equal:

        ```pycon
        >>> import torch
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import TorchTensorEqualityTester
        >>> config = EqualityConfig()
        >>> tester = TorchTensorEqualityTester()
        >>> # This example assumes CUDA is available
        >>> tester.objects_are_equal(
        ...     torch.ones(2, 3),
        ...     torch.ones(2, 3).cuda() if torch.cuda.is_available() else torch.ones(2, 3),
        ...     config,
        ... )  # doctest: +SKIP
        False

        ```
    """

    def __init__(self) -> None:
        """Initialize the PyTorch tensor equality tester.

        The handler chain performs checks in this order:
        1. SameObjectHandler: Quick check for object identity
        2. SameTypeHandler: Verify both are torch tensors
        3. SameDTypeHandler: Ensure tensors have matching data types
        4. SameShapeHandler: Verify tensors have the same dimensions
        5. TorchTensorSameDeviceHandler: Check tensors are on the same device
        6. TorchTensorEqualHandler: Element-wise equality with tolerance

        Raises:
            RuntimeError: If PyTorch is not installed.
        """
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
        actual: torch.Tensor,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)
