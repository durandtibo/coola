r"""Implement equality testers for PyTorch tensors and packed sequences.

This module provides equality testers for torch.Tensor and
torch.nn.utils.rnn.PackedSequence with support for device, dtype, shape,
and value comparisons including NaN and tolerance handling.
"""

from __future__ import annotations

__all__ = ["TorchPackedSequenceEqualityTester", "TorchTensorEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality._helpers.format import format_value_difference
from coola.equality._helpers.torch import tensor_close, tensor_equal
from coola.equality._helpers.utils import AttributeName
from coola.equality.tester._array import check_dtype_and_shape
from coola.equality.tester.base import BaseEqualityTester
from coola.utils.imports import check_torch, is_torch_available

if TYPE_CHECKING or is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine


class TorchPackedSequenceEqualityTester(BaseEqualityTester[torch.nn.utils.rnn.PackedSequence]):
    r"""Implement an equality tester for
    ``torch.nn.utils.rnn.PackedSequence``.

    This tester compares the data tensor and the metadata of two packed
    sequences: ``batch_sizes``, ``sorted_indices`` and
    ``unsorted_indices``. The location of a difference is the name of
    the attribute.

    Raises:
        MissingPackageError: If ``torch`` is not installed.

    Example:
        ```pycon
        >>> import torch
        >>> from torch.nn.utils.rnn import pack_padded_sequence
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> def make(value):
        ...     return pack_padded_sequence(
        ...         torch.tensor([[1.0, value], [3.0, 4.0]]),
        ...         lengths=torch.tensor([2, 1]),
        ...         batch_first=True,
        ...         enforce_sorted=False,
        ...     )
        ...
        >>> engine.run(make(2.0), make(2.0))
        True
        >>> engine.run(make(2.0), make(5.0))
        False

        ```
    """

    def __init__(self) -> None:
        check_torch()

    def objects_are_equal(
        self,
        actual: torch.nn.utils.rnn.PackedSequence,
        expected: torch.nn.utils.rnn.PackedSequence,
        _ctx: Engine,
    ) -> Iterator[tuple[object, object, object]]:
        r"""Test two packed sequences.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            _ctx: The ``Engine`` of the comparison.

        Returns:
            The pairs of ``data``, ``batch_sizes``, ``sorted_indices``
                and ``unsorted_indices`` that must all be equal.
        """
        return iter(
            (getattr(actual, name), getattr(expected, name), AttributeName(name))
            for name in ("data", "batch_sizes", "sorted_indices", "unsorted_indices")
        )


class TorchTensorEqualityTester(BaseEqualityTester[torch.Tensor]):
    r"""Implement an equality tester for ``torch.Tensor``.

    Two tensors are equal if they have the same data type, shape,
    device, layout and equal elements. The numbers are compared with
    the tolerance and ``equal_nan`` of the comparison, and a boolean
    tensor is exact. The tensors on the ``meta`` device have no data:
    only the metadata is compared. The location of a difference is the
    index of the first element that differs.

    Raises:
        MissingPackageError: If ``torch`` is not installed.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.equality import compare
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import TorchTensorEqualityTester, get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> tester = TorchTensorEqualityTester()
        >>> tester.objects_are_equal(torch.ones(2, 3), torch.ones(2, 3), engine)
        True
        >>> tester.objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3), engine)
        False
        >>> compare(torch.ones(2, 3), torch.zeros(2, 3)).path
        (0, 0)

        ```
    """

    def __init__(self) -> None:
        check_torch()

    def objects_are_equal(self, actual: torch.Tensor, expected: torch.Tensor, ctx: Engine) -> bool:
        r"""Test two tensors.

        The tensors must have the same dtype, shape, device and layout;
        the elements are compared with the tolerance and ``equal_nan``.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the objects are equal, otherwise ``False`` (the
                reason is recorded in the engine).
        """
        failure = check_dtype_and_shape(actual, expected, ctx)
        if failure is not None:
            return failure
        if actual.device != expected.device:
            return ctx.fail(
                lambda: (
                    f"torch.Tensors have different devices: {actual.device} vs {expected.device}"
                )
            )
        if actual.layout != expected.layout:
            return ctx.fail(
                lambda: (
                    f"torch.Tensors have different layouts: {actual.layout} vs {expected.layout}"
                )
            )
        if tensor_equal(actual, expected, ctx.config):
            return True
        if not ctx.explaining:
            return False
        # The index is looked up only when the difference is explained.
        close = tensor_close(actual, expected, ctx.config)
        index = tuple((~close).nonzero()[0].tolist())
        dense1 = actual if actual.layout == torch.strided else actual.to_dense()
        dense2 = expected if expected.layout == torch.strided else expected.to_dense()
        return ctx.fail(
            lambda: format_value_difference(dense1[index], dense2[index], name="values"), *index
        )


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the PyTorch objects.

    Returns:
        The equality testers of the PyTorch objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the PyTorch objects.
    """
    if not is_torch_available():
        return {}

    return {
        torch.nn.utils.rnn.PackedSequence: TorchPackedSequenceEqualityTester(),
        torch.Tensor: TorchTensorEqualityTester(),
    }
