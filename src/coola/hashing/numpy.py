r"""Provide deterministic hashing for NumPy arrays."""

from __future__ import annotations

__all__ = ["NumpyArrayHasher", "hash_array_content"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.bytes import hash_bytes
from coola.hashing.repr import ReprHasher
from coola.hashing.string import hash_tagged
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from coola.hashing.base import Hasher
    from coola.hashing.registry import HasherRegistry


class NumpyArrayHasher(InlineDisplayMixin, BaseHasher["np.ndarray"]):
    r"""Hasher for ``numpy.ndarray`` objects.

    The hash depends on the dtype (with its byte order), the shape and
    the values, not on the memory layout: a non-contiguous view hashes
    like its contiguous copy. ``-0.0`` hashes like ``0.0``, like in the
    equality. The items of an ``object`` array are hashed through the
    registry. For a masked array, the mask is part of the hash and the
    masked values are not.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.hashing import HasherRegistry, NumpyArrayHasher
        >>> hasher = NumpyArrayHasher()
        >>> hasher
        NumpyArrayHasher()
        >>> len(hasher.hash(np.arange(3), registry=HasherRegistry()))
        64

        ```
    """

    def hash(
        self,
        data: np.ndarray,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        tag = "numpy.ndarray"
        if isinstance(data, np.ma.MaskedArray):
            tag = "numpy.ma.MaskedArray"
            mask = np.ma.getmaskarray(data)
            mask_hash = hash_bytes(mask.tobytes(), length=length)
            data = np.where(mask, np.zeros_like(data.data), data.data)
        else:
            mask_hash = ""
        if data.dtype.kind == "O":
            items = [
                registry.hash(x, length=length, ignore_unhashable=ignore_unhashable)
                for x in data.ravel().tolist()
            ]
            raw_hash = hash_tagged("items", "".join(items), length=length)
        else:
            data = canonicalize_floats(data)
            raw_hash = hash_bytes(np.ascontiguousarray(data).tobytes(), length=length)
        return hash_array_content(
            tag, str(data.dtype.str), data.shape, raw_hash + mask_hash, length=length
        )


def canonicalize_floats(data: np.ndarray) -> np.ndarray:
    r"""Return an array where the floating-point values have one bit
    pattern per value.

    ``-0.0`` becomes ``0.0`` and every NaN becomes the default NaN, so
    equal values (and NaNs, which are equal for ``equal_nan``) have the
    same bytes whatever the sign or the payload of the NaN. The real and
    the imaginary parts of a complex array are processed separately. The
    other dtypes are returned unchanged.

    Args:
        data: The array to canonicalize. It is not modified.

    Returns:
        The canonicalized array.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.hashing.numpy import canonicalize_floats
        >>> canonicalize_floats(np.array([-0.0, -np.nan])).tobytes() == (
        ...     np.array([0.0, np.nan]).tobytes()
        ... )
        True

        ```
    """
    kind = data.dtype.kind
    if kind == "f":
        return np.where(np.isnan(data), np.array(np.nan, dtype=data.dtype), data + 0)
    if kind == "c":
        out = np.empty_like(data)
        out.real = canonicalize_floats(data.real)
        out.imag = canonicalize_floats(data.imag)
        return out
    return data


def hash_array_content(
    tag: str, dtype: str, shape: tuple[int, ...], content_hash: str, length: int = 64
) -> str:
    r"""Combine the description of an array and the hash of its values.

    Args:
        tag: The type tag.
        dtype: The string form of the data type.
        shape: The shape of the array.
        content_hash: The hash of the values.
        length: The length of the returned hash.

    Returns:
        The hash of the array.

    Raises:
        ValueError: If ``length`` is not an even number between 2 and
            128.
    """
    return hash_tagged(tag, f"{dtype}\0{tuple(shape)}\0{content_hash}", length=length)


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the NumPy arrays and scalars.

    Returns:
        The hashers of the NumPy arrays and scalars and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the NumPy arrays and scalars.
    """
    if not is_numpy_available():
        return {}
    # The NumPy scalars are hashed like the equivalent builtin values.
    hasher = ReprHasher()
    return {
        np.ndarray: NumpyArrayHasher(),
        np.bool_: hasher,
        np.integer: hasher,
        np.floating: hasher,
        np.complexfloating: hasher,
    }
