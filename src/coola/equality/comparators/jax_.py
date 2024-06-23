r"""Implement an equality comparator for ``jax.numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["JaxArrayEqualityComparator", "get_type_comparator_mapping"]

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.handlers import (
    JaxArrayEqualHandler,
    SameDTypeHandler,
    SameObjectHandler,
    SameShapeHandler,
    SameTypeHandler,
)
from coola.utils.imports import check_jax, is_jax_available

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

logger = logging.getLogger(__name__)


class JaxArrayEqualityComparator(BaseEqualityComparator[jnp.ndarray]):
    r"""Implement an equality comparator for ``jax.numpy.ndarray``.

    Example usage:

    ```pycon

    >>> import jax.numpy as jnp
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.comparators import JaxArrayEqualityComparator
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> comparator = JaxArrayEqualityComparator()
    >>> comparator.equal(jnp.ones((2, 3)), jnp.ones((2, 3)), config)
    True
    >>> comparator.equal(jnp.ones((2, 3)), jnp.zeros((2, 3)), config)
    False

    ```
    """

    def __init__(self) -> None:
        check_jax()
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(SameDTypeHandler()).chain(
            SameShapeHandler()
        ).chain(JaxArrayEqualHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> JaxArrayEqualityComparator:
        return self.__class__()

    def equal(self, actual: jnp.ndarray, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    This function returns an empty dictionary if ``jax`` is not
    installed.

    Returns:
        The mapping between the types and the equality comparators.

    Example usage:

    ```pycon

    >>> from coola.equality.comparators.jax_ import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'jax.Array'>: JaxArrayEqualityComparator(),
     <class 'jaxlib.xla_extension.ArrayImpl'>: JaxArrayEqualityComparator()}

    ```
    """
    if not is_jax_available():
        return {}
    cmp = JaxArrayEqualityComparator()
    return {jnp.ndarray: cmp, get_array_impl_class(): cmp}


@lru_cache(maxsize=1)
def get_array_impl_class() -> type:
    r"""Get the array implementation class.

    Returns:
        The array implementation class.
    """
    return jnp.ones(1).__class__
