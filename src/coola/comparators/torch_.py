r"""Implement some equality operators for ``torch.Tensor`` and
``torch.nn.utils.rnn.PackedSequence``.

The operators are registered only if ``torch`` is available.
"""

from __future__ import annotations

__all__ = [
    "PackedSequenceAllCloseOperator",
    "PackedSequenceEqualityOperator",
    "TensorAllCloseOperator",
    "TensorEqualityOperator",
    "get_mapping_allclose",
    "get_mapping_equality",
]

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.comparators.base import BaseAllCloseOperator, BaseEqualityOperator
from coola.utils.imports import check_torch, is_torch_available

if TYPE_CHECKING:
    from coola.testers import BaseAllCloseTester, BaseEqualityTester

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

logger = logging.getLogger(__name__)


class PackedSequenceAllCloseOperator(BaseAllCloseOperator[torch.nn.utils.rnn.PackedSequence]):
    r"""Implement an allclose operator for
    ``torch.nn.utils.rnn.PackedSequence``.

    Example usage:

    ```pycon
    >>> import torch
    >>> from coola.comparators import PackedSequenceAllCloseOperator
    >>> from coola.testers import AllCloseTester
    >>> tester = AllCloseTester()
    >>> op = PackedSequenceAllCloseOperator()
    >>> op.allclose(
    ...     tester,
    ...     torch.nn.utils.rnn.pack_padded_sequence(
    ...         input=torch.arange(10, dtype=torch.float).view(2, 5),
    ...         lengths=torch.tensor([5, 3], dtype=torch.long),
    ...         batch_first=True,
    ...     ),
    ...     torch.nn.utils.rnn.pack_padded_sequence(
    ...         input=torch.arange(10, dtype=torch.float).view(2, 5),
    ...         lengths=torch.tensor([5, 3], dtype=torch.long),
    ...         batch_first=True,
    ...     ),
    ... )
    True

    ```
    """

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> PackedSequenceAllCloseOperator:
        return self.__class__()

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: torch.nn.utils.rnn.PackedSequence,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, torch.nn.utils.rnn.PackedSequence):
            if show_difference:
                logger.info(
                    f"object2 is not a `torch.nn.utils.rnn.PackedSequence`: {type(object2)}"
                )
            return False
        object_equal = (
            tester.allclose(
                object1=object1.data,
                object2=object2.data,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                show_difference=show_difference,
            )
            and tester.allclose(
                object1=object1.batch_sizes,
                object2=object2.batch_sizes,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                show_difference=show_difference,
            )
            and tester.allclose(
                object1=object1.sorted_indices,
                object2=object2.sorted_indices,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                show_difference=show_difference,
            )
            and tester.allclose(
                object1=object1.unsorted_indices,
                object2=object2.unsorted_indices,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                show_difference=show_difference,
            )
        )
        if show_difference and not object_equal:
            logger.info(
                f"`torch.nn.utils.rnn.PackedSequence` are different\nobject1=\n{object1}\n"
                f"object2=\n{object2}"
            )
        return object_equal


class PackedSequenceEqualityOperator(BaseEqualityOperator[torch.nn.utils.rnn.PackedSequence]):
    r"""Implement an equality operator for
    ``torch.nn.utils.rnn.PackedSequence``.

    Example usage:

    ```pycon
    >>> import torch
    >>> from coola.comparators import PackedSequenceEqualityOperator
    >>> from coola.testers import EqualityTester
    >>> tester = EqualityTester()
    >>> op = PackedSequenceEqualityOperator()
    >>> op.equal(
    ...     tester,
    ...     torch.nn.utils.rnn.pack_padded_sequence(
    ...         input=torch.arange(10, dtype=torch.float).view(2, 5),
    ...         lengths=torch.tensor([5, 3], dtype=torch.long),
    ...         batch_first=True,
    ...     ),
    ...     torch.nn.utils.rnn.pack_padded_sequence(
    ...         input=torch.arange(10, dtype=torch.float).view(2, 5),
    ...         lengths=torch.tensor([5, 3], dtype=torch.long),
    ...         batch_first=True,
    ...     ),
    ... )
    True

    ```
    """

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> PackedSequenceEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: torch.nn.utils.rnn.PackedSequence,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, torch.nn.utils.rnn.PackedSequence):
            if show_difference:
                logger.info(
                    f"object2 is not a `torch.nn.utils.rnn.PackedSequence`: {type(object2)}"
                )
            return False
        object_equal = (
            tester.equal(object1.data, object2.data, show_difference)
            and tester.equal(object1.batch_sizes, object2.batch_sizes, show_difference)
            and tester.equal(object1.sorted_indices, object2.sorted_indices, show_difference)
            and tester.equal(object1.unsorted_indices, object2.unsorted_indices, show_difference)
        )
        if show_difference and not object_equal:
            logger.info(
                f"`torch.nn.utils.rnn.PackedSequence` are different\nobject1=\n{object1}\n"
                f"object2=\n{object2}"
            )
        return object_equal


class TensorAllCloseOperator(BaseAllCloseOperator[torch.Tensor]):
    r"""Implement an allclose operator for ``torch.Tensor``.

    Example usage:

    ```pycon
    >>> import torch
    >>> from coola.comparators import TensorAllCloseOperator
    >>> from coola.testers import AllCloseTester
    >>> tester = AllCloseTester()
    >>> op = TensorAllCloseOperator()
    >>> op.allclose(tester, torch.arange(21), torch.arange(21))
    True

    ```
    """

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> TensorAllCloseOperator:
        return self.__class__()

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: torch.Tensor,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, torch.Tensor):
            if show_difference:
                logger.info(f"object2 is not a torch.Tensor: {type(object2)}")
            return False
        if object1.dtype != object2.dtype:
            if show_difference:
                logger.info(
                    f"torch.Tensor data types are different: {object1.shape} vs {object2.shape}"
                )
            return False
        if object1.device != object2.device:
            if show_difference:
                logger.info(
                    f"torch.Tensor devices are different: {object1.device} vs {object2.device}"
                )
            return False
        if object1.shape != object2.shape:
            if show_difference:
                logger.info(
                    f"torch.Tensor shapes are different: {object1.shape} vs {object2.shape}"
                )
            return False
        object_equal = object1.allclose(object2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if show_difference and not object_equal:
            logger.info(f"torch.Tensors are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


class TensorEqualityOperator(BaseEqualityOperator[torch.Tensor]):
    r"""Implement an equality operator for ``torch.Tensor``.

    Example usage:

    ```pycon
    >>> import torch
    >>> from coola.comparators import TensorEqualityOperator
    >>> from coola.testers import EqualityTester
    >>> tester = EqualityTester()
    >>> op = TensorEqualityOperator()
    >>> op.equal(tester, torch.arange(21), torch.arange(21))
    True

    ```
    """

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> TensorEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: torch.Tensor,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, torch.Tensor):
            if show_difference:
                logger.info(f"object2 is not a torch.Tensor: {type(object2)}")
            return False
        if object1.dtype != object2.dtype:
            if show_difference:
                logger.info(
                    f"torch.Tensor data types are different: {object1.dtype} vs {object2.dtype}"
                )
            return False
        if object1.device != object2.device:
            if show_difference:
                logger.info(
                    f"torch.Tensor devices are different: {object1.device} vs {object2.device}"
                )
            return False
        object_equal = object1.equal(object2)
        if show_difference and not object_equal:
            logger.info(f"torch.Tensors are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Get a default mapping between the types and the allclose
    operators.

    This function returns an empty dictionary if torch is not
    installed.

    Returns:
        The mapping between the types and the allclose operators.

    Example usage:

    ```pycon
    >>> from coola.comparators.torch_ import get_mapping_allclose
    >>> get_mapping_allclose()
    {<class 'torch.Tensor'>: TensorAllCloseOperator(),
     <class 'torch.nn.utils.rnn.PackedSequence'>: PackedSequenceAllCloseOperator()}

    ```
    """
    if not is_torch_available():
        return {}
    return {
        torch.Tensor: TensorAllCloseOperator(),
        torch.nn.utils.rnn.PackedSequence: PackedSequenceAllCloseOperator(),
    }


def get_mapping_equality() -> dict[type[object], BaseEqualityOperator]:
    r"""Get a default mapping between the types and the equality
    operators.

    This function returns an empty dictionary if torch is not
    installed.

    Returns:
        The mapping between the types and the equality operators.

    Example usage:

    ```pycon
    >>> from coola.comparators.torch_ import get_mapping_equality
    >>> get_mapping_equality()
    {<class 'torch.Tensor'>: TensorEqualityOperator(),
     <class 'torch.nn.utils.rnn.PackedSequence'>: PackedSequenceEqualityOperator()}

    ```
    """
    if not is_torch_available():
        return {}
    return {
        torch.Tensor: TensorEqualityOperator(),
        torch.nn.utils.rnn.PackedSequence: PackedSequenceEqualityOperator(),
    }