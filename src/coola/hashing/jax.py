r"""Provide deterministic hashing for JAX arrays."""

from __future__ import annotations

__all__ = ["JaxArrayHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.bytes import hash_bytes
from coola.hashing.numpy import hash_array_content
from coola.utils.imports import is_jax_available, is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING or is_jax_available():
    import jax
else:  # pragma: no cover
    from coola.utils.fallback.jax import jax

if TYPE_CHECKING:
    from coola.hashing.base import Hasher
    from coola.hashing.registry import HasherRegistry


class JaxArrayHasher(InlineDisplayMixin, BaseHasher["jax.Array"]):
    r"""Hasher for ``jax.Array`` objects.

    The hash depends on the dtype, the shape and the values, like
    ``NumpyArrayHasher``, but a JAX array and a NumPy array with the
    same content hash differently. ``-0.0`` hashes like ``0.0``.

    Example:
        ```pycon
        >>> import jax.numpy as jnp
        >>> from coola.hashing import HasherRegistry, JaxArrayHasher
        >>> hasher = JaxArrayHasher()
        >>> hasher
        JaxArrayHasher()
        >>> len(hasher.hash(jnp.arange(3), registry=HasherRegistry()))
        64

        ```
    """

    def hash(
        self,
        data: jax.Array,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        array = np.asarray(data)
        if array.dtype.kind in "fc":
            array = array + 0
        return hash_array_content(
            "jax.Array",
            str(array.dtype),
            array.shape,
            hash_bytes(np.ascontiguousarray(array).tobytes(), length=length),
            length=length,
        )


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the JAX arrays.

    Returns:
        The hashers of the JAX arrays and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the JAX arrays.
    """
    if not is_jax_available():
        return {}
    return {jax.Array: JaxArrayHasher()}
