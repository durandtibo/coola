r"""Provide deterministic hashing for NumPy arrays."""

from __future__ import annotations

__all__ = ["NumpyArrayHasher", "hash_array_content"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node, digest, tagged
from coola.hashing.scalar import ReprHasher
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING:
    from coola.hashing.base import Hasher


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
        >>> hasher.encode(np.arange(3)).startswith(b"numpy.ndarray")
        True

        ```
    """

    def encode(self, data: np.ndarray) -> Encoding:
        r"""Describe a NumPy array.

        A numeric array is a leaf: the dtype, the shape and the digest of
        the canonical values (and of the mask for a masked array). An
        ``object`` array is a ``Node`` whose children are the dtype, the
        shape and the items, which the engine hashes through the registry.

        Args:
            data: The array to describe.

        Returns:
            The leaf or the ``Node`` that describes ``data``.
        """
        tag = "numpy.ndarray"
        mask = None
        if isinstance(data, np.ma.MaskedArray):
            tag = "numpy.ma.MaskedArray"
            mask = np.ma.getmaskarray(data).tobytes()
            data = np.where(np.ma.getmaskarray(data), np.zeros_like(data.data), data.data)
        dtype = _dtype_str(data.dtype)
        if data.dtype.kind == "O":
            children = [dtype, data.shape, data.ravel().tolist()]
            if mask is not None:
                children.append(mask)
            return Node(tag, children)
        data = canonicalize_floats(data)
        content = digest(np.ascontiguousarray(data).tobytes())
        if mask is not None:
            content += digest(mask)
        return hash_array_content(tag, dtype, data.shape, content)


def _dtype_str(dtype: np.dtype) -> str:
    r"""Return a string that identifies a data type.

    ``dtype.str`` is ``'|V8'`` for every structured dtype of 8 bytes, so
    the field names, formats and padding (``descr``) are used for them.

    Args:
        dtype: The data type.

    Returns:
        The string that identifies the data type.
    """
    return str(dtype.descr) if dtype.names else str(dtype.str)


def _is_ml_dtypes_float(dtype: np.dtype) -> bool:
    r"""Indicate if a dtype is a floating-point type of ``ml_dtypes``
    (``bfloat16``, ``float8_*``, ...) with the kind ``V``.

    Args:
        dtype: The data type.

    Returns:
        ``True`` if the data type comes from ``ml_dtypes``, otherwise
            ``False``.
    """
    return dtype.type.__module__.startswith("ml_dtypes") and dtype.name.startswith(
        ("bfloat", "float")
    )


def canonicalize_floats(data: np.ndarray) -> np.ndarray:
    r"""Return an array where the floating-point values have one bit
    pattern per value.

    The fields of a structured array are processed one by one. ``-0.0`` becomes ``0.0`` and every NaN becomes the default NaN, so
    equal values (and NaNs, which are equal for ``equal_nan``) have the
    same bytes whatever the sign or the payload of the NaN. The real and
    the imaginary parts of a complex array are processed separately. The
    ``ml_dtypes`` floating-point types (``bfloat16``, ``float8_*``) are
    processed through ``float32``. The other dtypes are returned unchanged.

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
    names = data.dtype.names
    if names:
        # A structured array: its float fields are canonicalized one by one, on a copy.
        data = data.copy()
        for name in names:
            data[name] = canonicalize_floats(data[name])
        return data
    kind = data.dtype.kind
    if kind == "V" and _is_ml_dtypes_float(data.dtype):
        # ``bfloat16`` and ``float8_*`` have the kind ``V``. The conversion to
        # ``float32`` is exact, and so is the conversion back.
        return canonicalize_floats(data.astype(np.float32)).astype(data.dtype)
    if kind == "f":
        return np.where(np.isnan(data), np.array(np.nan, dtype=data.dtype), data + 0)
    if kind == "c":
        out = np.empty_like(data)
        out.real = canonicalize_floats(data.real)
        out.imag = canonicalize_floats(data.imag)
        return out
    return data


def hash_array_content(tag: str, dtype: str, shape: tuple[int, ...], content: bytes) -> bytes:
    r"""Combine the description of an array and the digest of its values.

    Args:
        tag: The type tag.
        dtype: The string form of the data type.
        shape: The shape of the array.
        content: The digest of the values (``digest(raw_bytes)``).

    Returns:
        The leaf that describes the array.
    """
    return tagged(tag, f"{dtype}\0{tuple(shape)}\0".encode() + content)


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
