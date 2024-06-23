r"""Implement an equality comparator for ``torch.Tensor``s and
``torch.nn.utils.rnn.PackedSequence``s."""

from __future__ import annotations

__all__ = [
    "TorchPackedSequenceEqualityComparator",
    "TorchTensorEqualityComparator",
    "get_type_comparator_mapping",
]

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    SameAttributeHandler,
    SameDataHandler,
    SameDTypeHandler,
    SameObjectHandler,
    SameShapeHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.equality.handlers.torch_ import (
    TorchTensorEqualHandler,
    TorchTensorSameDeviceHandler,
)
from coola.utils import check_torch, is_torch_available

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class TorchPackedSequenceEqualityComparator(
    BaseEqualityComparator[torch.nn.utils.rnn.PackedSequence]
):
    r"""Implement an equality comparator for ``torch.Tensor``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import TorchPackedSequenceEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = TorchTensorEqualityComparator()
    >>> comparator.equal(torch.ones(2, 3), torch.ones(2, 3), config)
    True
    >>> comparator.equal(torch.ones(2, 3), torch.zeros(2, 3), config)
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
        ).chain(
            TrueHandler()
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> TorchPackedSequenceEqualityComparator:
        return self.__class__()

    def equal(
        self, actual: torch.nn.utils.rnn.PackedSequence, expected: Any, config: EqualityConfig
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)


class TorchTensorEqualityComparator(BaseEqualityComparator[torch.Tensor]):
    r"""Implement an equality comparator for ``torch.Tensor``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import TorchTensorEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = TorchTensorEqualityComparator()
    >>> comparator.equal(torch.ones(2, 3), torch.ones(2, 3), config)
    True
    >>> comparator.equal(torch.ones(2, 3), torch.zeros(2, 3), config)
    False

    ```
    """

    def __init__(self) -> None:
        check_torch()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameDTypeHandler()).chain(
            SameShapeHandler()
        ).chain(TorchTensorSameDeviceHandler()).chain(TorchTensorEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> TorchTensorEqualityComparator:
        return self.__class__()

    def equal(self, actual: torch.Tensor, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    This function returns an empty dictionary if ``torch`` is not
    installed.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon

    >>> from coola.equality.comparators.torch_ import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'torch.nn.utils.rnn.PackedSequence'>: TorchPackedSequenceEqualityComparator(),
     <class 'torch.Tensor'>: TorchTensorEqualityComparator()}

    ```
    """
    if not is_torch_available():
        return {}
    return {
        torch.nn.utils.rnn.PackedSequence: TorchPackedSequenceEqualityComparator(),
        torch.Tensor: TorchTensorEqualityComparator(),
    }
