from __future__ import annotations

__all__ = [
    "JaxArrayAllCloseOperator",
    "JaxArrayEqualityOperator",
    "get_mapping_allclose",
    "get_mapping_equality",
]

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from coola.comparators.base import BaseAllCloseOperator, BaseEqualityOperator
from coola.utils.imports import check_jax, is_jax_available

if TYPE_CHECKING:
    from coola.testers import BaseAllCloseTester, BaseEqualityTester

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()

logger = logging.getLogger(__name__)


class JaxArrayAllCloseOperator(BaseAllCloseOperator[jnp.ndarray]):
    r"""Implements an allclose operator for
    ``jax.Array``/``jax.numpy.ndarray``.

    Args:
    ----
        check_dtype (bool, optional): If ``True``, the data type of
            the arrays are checked, otherwise the data types are
            ignored. Default: ``True``
    """

    def __init__(self, check_dtype: bool = True) -> None:
        check_jax()
        self._check_dtype = bool(check_dtype)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._check_dtype == other._check_dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(check_dtype={self._check_dtype})"

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: jnp.ndarray,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, jnp.ndarray):
            if show_difference:
                logger.info(f"object2 is not a jax.numpy.ndarray: {type(object2)}")
            return False
        if self._check_dtype and object1.dtype != object2.dtype:
            if show_difference:
                logger.info(
                    f"jax.numpy.ndarray data types are different: {object1.shape} vs {object2.shape}"
                )
            return False
        if object1.shape != object2.shape:
            if show_difference:
                logger.info(
                    f"jax.numpy.ndarray shapes are different: {object1.shape} vs {object2.shape}"
                )
            return False
        object_equal = jnp.allclose(object1, object2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if show_difference and not object_equal:
            logger.info(
                f"jax.numpy.ndarrays are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal

    def clone(self) -> JaxArrayAllCloseOperator:
        return self.__class__(check_dtype=self._check_dtype)


class JaxArrayEqualityOperator(BaseEqualityOperator[jnp.ndarray]):
    r"""Implements an equality operator for
    ``jax.Array``/``jax.numpy.ndarray``.

    Args:
    ----
        check_dtype (bool, optional): If ``True``, the data type of
            the arrays are checked, otherwise the data types are
            ignored. Default: ``True``
    """

    def __init__(self, check_dtype: bool = True) -> None:
        check_jax()
        self._check_dtype = bool(check_dtype)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._check_dtype == other._check_dtype

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(check_dtype={self._check_dtype})"

    def clone(self) -> JaxArrayEqualityOperator:
        return self.__class__(check_dtype=self._check_dtype)

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: jnp.ndarray,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, jnp.ndarray):
            if show_difference:
                logger.info(f"object2 is not a jax.numpy.ndarray: {type(object2)}")
            return False
        if self._check_dtype and object1.dtype != object2.dtype:
            if show_difference:
                logger.info(
                    f"jax.numpy.ndarray data types are different: {object1.shape} vs {object2.shape}"
                )
            return False
        if object1.shape != object2.shape:
            if show_difference:
                logger.info(
                    f"jax.numpy.ndarray shapes are different: {object1.shape} vs {object2.shape}"
                )
            return False
        object_equal = jnp.array_equal(object1, object2)
        if show_difference and not object_equal:
            logger.info(
                f"jax.numpy.ndarrays are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Gets a default mapping between the types and the allclose
    operators.

    This function returns an empty dictionary if JAX is not
    installed.

    Returns:
    -------
        dict: The mapping between the types and the allclose
            operators.
    """
    if not is_jax_available():
        return {}
    return {
        jnp.ndarray: JaxArrayAllCloseOperator(),
        get_array_impl_class(): JaxArrayAllCloseOperator(),
    }


def get_mapping_equality() -> dict[type[object], BaseEqualityOperator]:
    r"""Gets a default mapping between the types and the equality
    operators.

    This function returns an empty dictionary if JAX is not
    installed.

    Returns:
    -------
        dict: The mapping between the types and the equality
            operators.
    """
    if not is_jax_available():
        return {}
    return {
        jnp.ndarray: JaxArrayEqualityOperator(),
        get_array_impl_class(): JaxArrayEqualityOperator(),
    }


@lru_cache(maxsize=1)
def get_array_impl_class() -> type:
    r"""Gets the array implementation class.

    Returns:
    -------
        The array implementation class.
    """
    return jnp.ones(1).__class__
