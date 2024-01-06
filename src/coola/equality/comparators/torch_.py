r"""Implement an equality comparator for ``torch.Tensor``s."""

from __future__ import annotations

__all__ = ["TensorEqualityComparator", "get_type_comparator_mapping"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    ArraySameDTypeHandler,
    ArraySameShapeHandler,
    SameObjectHandler,
    SameTypeHandler,
)
from coola.equality.handlers.torch_ import TensorEqualHandler, TensorSameDeviceHandler
from coola.utils import check_torch, is_torch_available

if is_torch_available():
    import torch

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class TensorEqualityComparator(BaseEqualityComparator[Any]):
    r"""Implement an equality comparator for ``torch.Tensor``.

    Example usage:

    ```pycon
    >>> import torch
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import TensorEqualityComparator
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = TensorEqualityComparator()
    >>> comparator.equal(torch.ones(2, 3), torch.ones(2, 3), config)
    True
    >>> comparator.equal(torch.ones(2, 3), torch.zeros(2, 3), config)
    False

    ```
    """

    def __init__(self) -> None:
        check_torch()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(ArraySameDTypeHandler()).chain(
            ArraySameShapeHandler()
        ).chain(TensorSameDeviceHandler()).chain(TensorEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> TensorEqualityComparator:
        return self.__class__()

    def equal(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(object1=object1, object2=object2, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    This function returns an empty dictionary if torch is not
    installed.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon
    >>> from coola.equality.comparators.torch_ import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'torch.Tensor'>: TensorEqualityComparator()}

    ```
    """
    if not is_torch_available():
        return {}
    return {torch.Tensor: TensorEqualityComparator()}
