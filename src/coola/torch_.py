from __future__ import annotations

import logging
from typing import Any

from coola.allclose import AllCloseTester, BaseAllCloseOperator, BaseAllCloseTester
from coola.equality import BaseEqualityOperator, BaseEqualityTester, EqualityTester
from coola.utils import check_torch, is_torch_available

if is_torch_available():
    from torch import Tensor, is_tensor
    from torch.nn.utils.rnn import PackedSequence
else:
    PackedSequence, Tensor = None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class PackedSequenceAllCloseOperator(BaseAllCloseOperator[PackedSequence]):
    r"""Implements an allclose operator for
    ``torch.nn.utils.rnn.PackedSequence``."""

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> PackedSequenceAllCloseOperator:
        return self.__class__()

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: PackedSequence,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, PackedSequence):
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


class PackedSequenceEqualityOperator(BaseEqualityOperator[PackedSequence]):
    r"""Implements an equality operator for
    ``torch.nn.utils.rnn.PackedSequence``."""

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> PackedSequenceEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: PackedSequence,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, PackedSequence):
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


class TensorAllCloseOperator(BaseAllCloseOperator[Tensor]):
    r"""Implements an allclose operator for ``torch.Tensor``."""

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> TensorAllCloseOperator:
        return self.__class__()

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Tensor,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not is_tensor(object2):
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


class TensorEqualityOperator(BaseEqualityOperator[Tensor]):
    r"""Implements an equality operator for ``torch.Tensor``."""

    def __init__(self) -> None:
        check_torch()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> TensorEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Tensor,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not is_tensor(object2):
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


if is_torch_available():  # pragma: no cover
    if not AllCloseTester.has_operator(PackedSequence):
        AllCloseTester.add_operator(PackedSequence, PackedSequenceAllCloseOperator())
    if not AllCloseTester.has_operator(Tensor):
        AllCloseTester.add_operator(Tensor, TensorAllCloseOperator())
    if not EqualityTester.has_operator(PackedSequence):
        EqualityTester.add_operator(PackedSequence, PackedSequenceEqualityOperator())
    if not EqualityTester.has_operator(Tensor):
        EqualityTester.add_operator(Tensor, TensorEqualityOperator())
