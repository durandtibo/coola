__all__ = [
    "PackedSequenceAllCloseOperator",
    "PackedSequenceEqualityOperator",
    "TensorAllCloseOperator",
    "TensorEqualityOperator",
]

import logging
from typing import Any

from coola.allclose import AllCloseTester, BaseAllCloseOperator, BaseAllCloseTester
from coola.equality import BaseEqualityOperator, BaseEqualityTester, EqualityTester
from coola.import_utils import is_torch_available

if is_torch_available():
    from torch import Tensor, is_tensor
    from torch.nn.utils.rnn import PackedSequence
else:
    PackedSequence, Tensor, is_tensor = None, None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class PackedSequenceAllCloseOperator(BaseAllCloseOperator[PackedSequence]):
    r"""Implements an allclose operator for
    ``torch.nn.utils.rnn.PackedSequence``."""

    def __init__(self):
        if not is_torch_available():
            raise RuntimeError(
                "`PackedSequenceAllCloseOperator` requires the `torch` package to be installed. "
                "You can install `torch` package with the command:\n\npip install torch\n"
            )

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

    def __init__(self):
        if not is_torch_available():
            raise RuntimeError(
                "`PackedSequenceEqualityOperator` requires the `torch` package to be installed. "
                "You can install `torch` package with the command:\n\npip install torch\n"
            )

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


class TensorAllCloseOperator(BaseAllCloseOperator[Tensor]):
    r"""Implements an allclose operator for ``torch.Tensor``."""

    def __init__(self):
        if not is_torch_available():
            raise RuntimeError(
                "`TensorAllCloseOperator` requires the `torch` package to be installed. You can "
                "install `torch` package with the command:\n\npip install torch\n"
            )

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

    def __init__(self):
        if not is_torch_available():
            raise RuntimeError(
                "`TensorEqualityOperator` requires the `torch` package to be installed. You can "
                "install `torch` package with the command:\n\npip install torch\n"
            )

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


if is_torch_available():
    if not AllCloseTester.has_allclose_operator(PackedSequence):
        AllCloseTester.add_allclose_operator(PackedSequence, PackedSequenceAllCloseOperator())
    if not AllCloseTester.has_allclose_operator(Tensor):
        AllCloseTester.add_allclose_operator(Tensor, TensorAllCloseOperator())
    if not EqualityTester.has_equality_operator(PackedSequence):
        EqualityTester.add_equality_operator(PackedSequence, PackedSequenceEqualityOperator())
    if not EqualityTester.has_equality_operator(Tensor):
        EqualityTester.add_equality_operator(Tensor, TensorEqualityOperator())
