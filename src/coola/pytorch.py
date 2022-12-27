__all__ = ["PackedSequenceEqualityOperator", "TensorEqualityOperator"]

import logging
from typing import Any

from coola import is_torch_available
from coola.equality import BaseEqualityOperator, BaseEqualityTester, EqualityTester

if is_torch_available():
    from torch import Tensor, is_tensor
    from torch.nn.utils.rnn import PackedSequence
else:
    PackedSequence, Tensor, is_tensor = None, None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class PackedSequenceEqualityOperator(BaseEqualityOperator[PackedSequence]):
    r"""Implements an equality operator for
    ``torch.nn.utils.rnn.PackedSequence``."""

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: PackedSequence,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
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


class TensorEqualityOperator(BaseEqualityOperator[Tensor]):
    r"""Implements an equality operator for ``torch.Tensor``."""

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Tensor,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
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
        object_equal = object1.equal(object2)
        if show_difference and not object_equal:
            logger.info(f"torch.Tensors are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


if is_torch_available() and not EqualityTester.has_equality_operator(Tensor):
    EqualityTester.add_equality_operator(PackedSequence, PackedSequenceEqualityOperator())
    EqualityTester.add_equality_operator(Tensor, TensorEqualityOperator())
