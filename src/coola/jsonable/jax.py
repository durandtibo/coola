r"""Define the transformers of the JAX arrays."""

from __future__ import annotations

__all__ = ["get_default_transformers"]

from typing import TYPE_CHECKING, Any

from coola.jsonable.transformers import ConvertTransformer
from coola.utils.imports import is_jax_available

if TYPE_CHECKING or is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    from coola.utils.fallback.jax import jnp

if TYPE_CHECKING:
    from coola.recursive.base import Transformer


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the JAX arrays.

    An array gives the nested lists of ``tolist``, which are converted in
    turn.

    Returns:
        The transformers of the JAX arrays and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the transformers of the JAX arrays.

    Example:
        ```pycon
        >>> from coola.jsonable.jax import get_default_transformers
        >>> [t.__name__ for t in get_default_transformers()]
        ['Array']

        ```
    """
    if not is_jax_available():
        return {}
    # ``jnp.ndarray`` is ``jax.Array``, which is in the MRO of the concrete array classes.
    return {jnp.ndarray: ConvertTransformer(lambda array: array.tolist())}
