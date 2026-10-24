r"""Implement handler to check if the objects have the same data."""

from __future__ import annotations

__all__ = ["SameDataHandler", "SupportsData"]

from typing import TYPE_CHECKING, Any, Protocol

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.mixin import HandlerEqualityMixin
from coola.equality.handler.utils import AttributeName, report_difference
from coola.utils.safe import safe_str

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class SupportsData(Protocol):
    r"""Implement a protocol to represent objects with a ``data``
    attribute.

    This protocol can be used to represent several objects like
    ``jax.numpy.ndarray``s, ``numpy.ndarray``s,  and
    ``torch.Tensor``s.
    """

    @property
    def data(self) -> Any:
        r"""The underlying data of the object."""
        return  # pragma: no cover


class SameDataHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects have the same data.

    This handler returns ``False`` if the two objects have different
    data, otherwise it passes the inputs to the next handler.
    The objects must have a ``data`` attribute (e.g. ``object.data``)
    which returns the data of the object.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import SameDataHandler, TrueHandler
        >>> config = EqualityConfig()
        >>> handler = SameDataHandler(next_handler=TrueHandler())
        >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
        True
        >>> handler.handle(np.ones((2, 3)), np.zeros((2, 3)), config)
        False

        ```
    """

    def handle(self, actual: SupportsData, expected: SupportsData, config: EqualityConfig) -> bool:
        if not config.registry.objects_are_equal(actual.data, expected.data, config):
            if config.show_difference:
                report_difference(
                    config,
                    "objects have different data: %s vs %s",
                    safe_str(actual.data),
                    safe_str(expected.data),
                    path=(AttributeName("data"),),
                )
            return False
        return self._handle_next(actual, expected, config=config)
