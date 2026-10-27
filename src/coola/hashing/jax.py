r"""Provide deterministic hashing for JAX arrays."""

from __future__ import annotations

__all__ = ["JaxArrayHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, digest
from coola.hashing.numpy import canonicalize_floats, hash_array_content
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
        >>> hasher.encode(jnp.arange(3)).startswith(b"jax.Array")
        True

        ```
    """

    def encode(self, data: jax.Array) -> Encoding:
        r"""Describe a JAX array.

        Args:
            data: The array to describe.

        Returns:
            The leaf that describes ``data``: the dtype, the shape and
            the digest of the canonical values.
        """
        array = canonicalize_floats(np.asarray(data))
        return hash_array_content(
            "jax.Array",
            str(array.dtype),
            array.shape,
            digest(np.ascontiguousarray(array).tobytes()),
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
