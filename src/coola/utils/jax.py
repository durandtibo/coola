r"""Implement some utility functions for ``jax.Array``s."""

from __future__ import annotations

__all__ = ["get_jax_array_types"]

from importlib import import_module

from coola.utils.imports import is_jax_available


def get_jax_array_types() -> tuple[type, ...]:
    r"""Get the types to register to handle the JAX arrays.

    ``jax.Array`` is in the MRO of the concrete array class in the recent
    versions of JAX, but not in the older ones (for example ``0.5``): the
    concrete class ``ArrayImpl`` has to be registered too. Creating an
    array to get this class would initialize the JAX backend as a side
    effect, so it is read from the module that defines it.

    Returns:
        ``jax.Array`` and, if it is not in the MRO of the concrete
            array class, the concrete array class. It is empty if JAX
            is not installed.

    Example:
        ```pycon
        >>> from coola.utils.jax import get_jax_array_types
        >>> get_jax_array_types()  # doctest: +SKIP
        (<class 'jax.Array'>,)

        ```
    """
    if not is_jax_available():
        return ()
    import jax  # noqa: PLC0415

    types: tuple[type, ...] = (jax.Array,)
    try:
        impl = import_module("jax._src.array").ArrayImpl
    except (ImportError, AttributeError):
        return types
    if jax.Array in impl.__mro__:
        return types
    return (*types, impl)
