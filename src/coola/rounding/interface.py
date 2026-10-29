r"""Define the public interface to round numerical values."""

from __future__ import annotations

__all__ = ["get_default_registry", "register_rounders", "round_values"]

import warnings
from typing import TYPE_CHECKING, Any

from coola.recursive.interface import recursive_apply
from coola.registry import TypeNotRegisteredError, TypeRegistry
from coola.registry.default import make_default_registry_singleton
from coola.rounding.common import (
    MissingPackagePolicy,
    Rounder,
    RoundMode,
    UnsupportedTypeError,
    validate_arguments,
)
from coola.rounding.native import get_default_rounders as get_native_rounders
from coola.utils.imports import MissingPackageError, package_available

if TYPE_CHECKING:
    from collections.abc import Mapping

    from coola.utils.singleton import LazySingleton

# (names of the library in the module of the types, module with the rounders)
_LAZY_ROUNDERS = (
    (("numpy",), "coola.rounding.numpy"),
    (("pandas",), "coola.rounding.pandas"),
    (("polars",), "coola.rounding.polars"),
    (("pyarrow",), "coola.rounding.pyarrow"),
    (("torch",), "coola.rounding.torch"),
)


def round_values(
    data: Any,
    *,
    mode: RoundMode = "round",
    decimals: int = 0,
    recursive: bool = False,
    on_missing_package: MissingPackagePolicy = "raise",
    registry: TypeRegistry[Rounder] | None = None,
) -> Any:
    r"""Round the numerical values of an object.

    The supported objects are the following ones, and a rounder can be
    registered for another type with ``register_rounders``:

    - the native ``int`` and ``float``;
    - the NumPy arrays and scalars;
    - the PyTorch tensors;
    - the pandas Series and DataFrames;
    - the polars Series and DataFrames;
    - the pyarrow Arrays, ChunkedArrays, Tables and RecordBatches.

    The integer values are never changed, whatever the mode and the
    number of decimals. A floating-point value stays a floating-point
    value of the same type (``round_values(2.5)`` is ``2.0``, not
    ``2``). ``NaN``, the infinite values and the missing values are
    kept. The input is not modified and the result never shares
    memory with it.

    The tables (pandas, polars and pyarrow) are heterogeneous: only
    their floating-point columns are rounded and the other columns
    (integers, strings, booleans, dates, ...) are left as they are. A
    Series, an array or a tensor must be numerical. The date and time
    types, the complex numbers and the other objects (``Decimal``,
    ``Fraction``, containers, ...) are not supported.

    Args:
        data: The object to round.
        mode: The rounding mode. ``"round"`` rounds to the nearest
            value and the halfway cases to the even digit,
            ``"floor"`` rounds towards
            negative infinity and ``"ceil"`` rounds towards positive
            infinity.
        decimals: The number of decimals to keep. It must be an
            integer (a NumPy integer is accepted) greater than or equal
            to ``0``.
        recursive: If ``True``, ``data`` can be a nested structure
            (``list``, ``tuple``, ``dict``, ``set``, dataclass, ...,
            as handled by ``coola.recursive.recursive_apply``) and
            every leaf is rounded. A leaf that is not supported (a
            string, ``None``, a date, ...) is kept as it is instead of
            raising an error. Defaults to ``False``.
        on_missing_package: What to do when the rounder of an object
            needs a package that is not installed (NumPy for the
            polars and pyarrow objects). ``"raise"`` raises an
            ``UnsupportedTypeError``, ``"warn"`` emits a
            ``UserWarning`` and returns the object unchanged, and
            ``"ignore"`` returns the object unchanged. With
            ``recursive=True`` the option applies to each leaf. An
            object with no rounder is not concerned by this option.
            Defaults to ``"raise"``.
        registry: The registry used to find the rounder of each type.
            If ``None``, the default registry is used.

    Returns:
        The object with the rounded values, with the same type, dtype
            and shape as ``data`` (a subclass of ``ndarray`` and a
            masked array keep their type and their mask).

    Raises:
        ValueError: If ``mode`` is not ``"round"``, ``"floor"`` or
            ``"ceil"``, if ``decimals`` is negative, or if
            ``on_missing_package`` is not ``"raise"``, ``"warn"`` or
            ``"ignore"``.
        TypeError: If ``decimals`` is not an integer.
        UnsupportedTypeError: If ``data`` is not supported (with
            ``recursive=False``), or if its rounder needs a package
            that is not installed and ``on_missing_package`` is
            ``"raise"``. It is a ``TypeError``.

    Notes:
        One rule is used for the three modes and for every library. A
        float is read as the decimal that ``repr`` shows (the shortest
        string that round-trips in its dtype). ``"floor"``, ``"ceil"``
        and ``"round"`` (halfway cases to the even digit) act on this
        decimal, and the result is the float of the dtype that is
        nearest to the grid point. So ``round_values(0.29,
        mode="floor", decimals=2)`` is ``0.29`` and not ``0.28``, and
        ``round_values(2.675, decimals=2)`` is ``2.68``.

        This differs from the Python ``round`` function and from
        ``numpy.round``, which round the exact binary value (Python)
        or scale and round (NumPy), so the halfway cases of a decimal
        literal give another result (``round(2.675, 2)`` is ``2.67``).

        The result equals ``Decimal(repr(x)).quantize(...)`` while
        ``abs(x) * 10**decimals`` is below ``1e13``. Above, it is
        within one grid step of it. A value that is zero after the
        rounding has the sign of the input (``-0.4`` ceiled is
        ``-0.0``, like ``numpy.ceil``).

    Example:
        ```pycon
        >>> from coola.rounding import round_values
        >>> round_values(2.567, decimals=1)
        2.6
        >>> round_values(2.567, mode="floor", decimals=2)
        2.56
        >>> round_values(2.567, mode="ceil")
        3.0
        >>> round_values({"a": [1.234, "x"], "b": (5.678, 3)}, decimals=1, recursive=True)
        {'a': [1.2, 'x'], 'b': (5.7, 3)}
        >>> import numpy as np
        >>> round_values(np.array([1.5, 2.5, -0.5]))
        array([ 2.,  2., -0.])
        >>> import torch
        >>> round_values(torch.tensor([1.234, 5.678]), mode="ceil", decimals=1)
        tensor([1.3000, 5.7000])
        >>> import pandas as pd
        >>> round_values(pd.DataFrame({"a": [1.234, 5.678], "b": [1, 2]}), decimals=2)
              a  b
        0  1.23  1
        1  5.68  2

        ```
    """
    decimals = validate_arguments(mode, decimals, on_missing_package)
    if registry is None:
        registry = get_default_registry()
    if recursive:
        return recursive_apply(
            data, lambda x: _round_leaf(x, registry, mode, decimals, on_missing_package)
        )
    return _round_object(data, registry, mode, decimals, on_missing_package)


def register_rounders(mapping: Mapping[type, Rounder], *, exist_ok: bool = False) -> None:
    r"""Register rounders into the default registry.

    A rounder is a function ``(data, mode, decimals) -> result`` that
    rounds the objects of one type. ``mode`` and ``decimals`` are
    validated by ``round_values``. The function returns a new object,
    and raises ``UnsupportedTypeError`` for a type or a dtype that it
    does not support. The rounder of the most specific type along the
    MRO of the object is used.

    Args:
        mapping: The mapping from types to rounders.
        exist_ok: If ``False``, raises an error if a type is already
            registered. Defaults to ``False``.

    Example:
        ```pycon
        >>> from coola.rounding import register_rounders, round_values
        >>> class Meters:
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> def round_meters(data, mode, decimals):
        ...     return Meters(round_values(data.value, mode=mode, decimals=decimals))
        ...
        >>> register_rounders({Meters: round_meters})
        >>> round_values(Meters(2.567), decimals=1).value
        2.6

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> TypeRegistry[Rounder]:
    r"""Return the default registry of the rounders.

    The registry is created on the first call and reused on all the
    subsequent calls (singleton pattern). It has the rounders of the
    native numbers, and the rounders of NumPy, PyTorch, pandas, polars
    and pyarrow, which are loaded the first time an object of the
    library is rounded. The plugins of the group
    ``coola.rounding.rounders`` are applied when it is created.

    Returns:
        The singleton registry.

    Example:
        ```pycon
        >>> from coola.rounding import get_default_registry
        >>> get_default_registry().has(float)
        True

        ```
    """
    return _default_registry.get()


def _round_leaf(
    data: Any,
    registry: TypeRegistry[Rounder],
    mode: RoundMode,
    decimals: int,
    on_missing_package: MissingPackagePolicy,
) -> Any:
    # Only an unsupported leaf is kept: the other errors are bugs and must not be hidden. A leaf
    # whose package is missing is not unsupported: with ``"raise"`` the error stops the walk.
    try:
        return _round_object(data, registry, mode, decimals, on_missing_package)
    except UnsupportedTypeError as exc:
        if isinstance(exc.__cause__, MissingPackageError):
            raise
        return data


def _round_object(
    data: Any,
    registry: TypeRegistry[Rounder],
    mode: RoundMode,
    decimals: int,
    on_missing_package: MissingPackagePolicy,
) -> Any:
    try:
        rounder = registry.resolve(type(data))
    except TypeNotRegisteredError as exc:
        msg = (
            f"Cannot round an object of type {type(data).__qualname__}; "
            "register a rounder with register_rounders"
        )
        raise UnsupportedTypeError(msg) from exc
    try:
        return rounder(data, mode, decimals)
    except MissingPackageError as exc:
        if on_missing_package == "raise":
            msg = (
                f"Cannot round an object of type {type(data).__qualname__}: "
                f"the package {exc.name!r} is not installed"
            )
            raise UnsupportedTypeError(msg) from exc
        if on_missing_package == "warn":
            msg = (
                f"The object of type {type(data).__qualname__} is not rounded: "
                f"the package {exc.name!r} is not installed"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        return data


def _register_default_rounders(registry: TypeRegistry[Rounder]) -> None:
    r"""Populate a registry with the default rounders.

    Args:
        registry: The registry to populate.
    """
    registry.register_many(get_native_rounders())
    # The rounders of the optional libraries are loaded the first time the registry needs a type
    # of the library, so the libraries are not imported with ``coola.rounding``.
    for names, module in _LAZY_ROUNDERS:
        if package_available(names[0]):
            registry.register_lazy(names, f"{module}:get_default_rounders")


_default_registry: LazySingleton[TypeRegistry[Rounder]] = make_default_registry_singleton(
    TypeRegistry, _register_default_rounders, plugin_group="coola.rounding.rounders"
)
