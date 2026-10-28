r"""Contain functions for polars DataFrame."""

from __future__ import annotations

__all__ = ["flatten_frame"]

import operator
from typing import TYPE_CHECKING, cast

from coola.utils.imports import is_polars_available
from coola.utils.version import compare_version
from coola.validation import validate_non_negative

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl


# The first versions of polars that have the options used below.
_UNNEST_SEPARATOR_VERSION = "1.35.0"  # ``DataFrame.unnest(separator=...)``
_EXPLODE_EMPTY_AS_NULL_VERSION = "1.36.0"  # ``DataFrame.explode(empty_as_null=...)``


def _is_nested_struct(dtype: pl.DataType) -> bool:
    r"""Return ``True`` if ``dtype`` is a ``List`` or ``Array`` whose
    inner type is a ``Struct``.

    Args:
        dtype: The polars dtype to inspect.

    Returns:
        ``True`` if ``dtype`` is a ``List``/``Array`` of ``Struct``,
            otherwise ``False``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.nested.polars import _is_nested_struct
        >>> _is_nested_struct(pl.List(pl.Struct({"x": pl.Int64})))
        True
        >>> _is_nested_struct(pl.List(pl.Int64))
        False

        ```
    """
    if isinstance(dtype, (pl.List, pl.Array)):
        return isinstance(dtype.inner, pl.Struct)
    return False


def _unnest_with_separator(frame: pl.DataFrame, columns: list[str], separator: str) -> pl.DataFrame:
    r"""Unnest struct columns with a separator prefix, compatible with
    polars>=1.0.0.

    On polars>=1.35.0 (the version that added the parameter), delegates to the native ``unnest(separator=...)`` parameter.
    On older versions, unnests without a separator and renames the resulting columns
    to ``<parent><separator><field>`` manually.

    Args:
        frame: A DataFrame containing the struct columns to unnest.
        columns: Names of the struct columns to unnest.
        separator: Separator used to build output column names following the
            ``<parent><separator><field>`` pattern.

    Returns:
        A DataFrame with the given struct columns unnested and output columns
        named ``<parent><separator><field>``.
    """
    if compare_version("polars", operator.ge, _UNNEST_SEPARATOR_VERSION):
        return frame.unnest(columns, separator=separator)
    rename_map = {
        field.name: f"{col}{separator}{field.name}"
        for col in columns
        for field in cast("pl.Struct", frame.schema[col]).fields
    }
    return frame.unnest(columns).rename(rename_map)


def _expand_list_columns(frame: pl.DataFrame, separator: str) -> pl.DataFrame:
    r"""Expand list and array columns into one column per index position.

    For each ``List`` or ``Array`` column, creates new columns named
    ``<col><separator><index>`` for each position up to the maximum list
    length across all rows. Rows with lists shorter than the maximum are
    filled with ``None``. The original list columns are dropped, except
    the columns where every list is null or empty: they have no position
    to expand, so they are kept as they are instead of being lost.

    Args:
        frame: A DataFrame that may contain list or array columns.
        separator: Separator used to build output column names following the
            ``<col><separator><index>`` pattern.

    Returns:
        A DataFrame with list and array columns replaced by
        their expanded positional columns. Non-list columns are unchanged.
        If no list or array columns are present the frame is returned as-is.

    Raises:
        ValueError: If a new column name is already used by a column of
            the frame or by another new column.
    """
    list_cols = [
        name for name, dtype in frame.schema.items() if isinstance(dtype, (pl.List, pl.Array))
    ]
    if not list_cols:
        return frame

    expansions: dict[str, pl.Series] = {}
    expanded_cols: list[str] = []
    for col in list_cols:
        series = frame[col]
        if isinstance(series.dtype, pl.Array):
            # The ``list`` namespace is not defined for the ``Array`` dtype.
            series = series.cast(pl.List(series.dtype.inner))
        max_len = cast("int", series.list.len().max() or 0)
        if max_len == 0:
            # Nothing to expand: dropping the column would lose it.
            continue
        expanded_cols.append(col)
        for i in range(max_len):
            name = f"{col}{separator}{i}"
            if name in frame.columns or name in expansions:
                msg = f"Cannot expand the list column {col!r}: the column {name!r} already exists"
                raise ValueError(msg)
            expansions[name] = series.list.get(i, null_on_oob=True)

    return frame.drop(expanded_cols).with_columns(
        pl.Series(name, values) for name, values in expansions.items()
    )


def _unnest_one_level(frame: pl.DataFrame, separator: str) -> pl.DataFrame:
    r"""Unnest all top-level struct columns in a DataFrame by one level.

    Columns that are lists or arrays of structs are exploded first to expose
    the inner structs, which are then unnested.

    Args:
        frame: A DataFrame that may contain struct columns.
        separator: Separator used to build output column names following the
            ``<parent><separator><field>`` pattern.

    Returns:
        A DataFrame with all top-level struct columns unnested
        by one level. Non-struct columns are passed through unchanged. If no
        struct columns are present the frame is returned as-is.
    """
    nested_struct_cols = [name for name, dtype in frame.schema.items() if _is_nested_struct(dtype)]
    if nested_struct_cols:
        try:
            frame = _explode(frame, nested_struct_cols)
        except pl.exceptions.ShapeError:
            # The lists of a row have different lengths, so they cannot be exploded in
            # parallel: explode them one after the other (cross product of the lists).
            for name in nested_struct_cols:
                frame = _explode(frame, [name])

    struct_cols = [name for name, dtype in frame.schema.items() if isinstance(dtype, pl.Struct)]
    if not struct_cols:
        return frame

    return _unnest_with_separator(frame, struct_cols, separator)


def _explode(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    r"""Explode the given columns, keeping the empty lists as null.

    Args:
        frame: The DataFrame to explode.
        columns: The names of the list columns to explode.

    Returns:
        The exploded DataFrame.
    """
    if compare_version("polars", operator.ge, _EXPLODE_EMPTY_AS_NULL_VERSION):
        return frame.explode(columns, empty_as_null=True)
    # empty_as_null is not available in older polars versions
    return frame.explode(columns)


def flatten_frame(
    frame: pl.DataFrame,
    separator: str = ".",
    depth: int | None = None,
    expand_lists: bool = False,
) -> pl.DataFrame:
    r"""Recursively unnest struct columns and optionally expand
    list/array columns.

    Non-nested columns are passed through unchanged. Struct columns are expanded level by level up
    to ``depth`` levels. Columns that are lists or arrays of structs are
    automatically exploded to expose their inner structs before unnesting.
    Several such columns are exploded together if their lists have the same
    length in every row, otherwise one after the other, which gives one row
    for each combination of their elements.
    If ``expand_lists`` is enabled, remaining ``List`` and ``Array`` columns
    are expanded into one column per index position following the
    ``<col><separator><index>`` pattern, filled with ``None`` where the list
    is shorter than the maximum length.

    Args:
        frame: A DataFrame that may contain struct, list, or array
            columns at any nesting level.
        separator: Separator used to build output column names following the
            ``<parent><separator><field>`` pattern. Defaults to ``"."``.
        depth: Maximum number of struct nesting levels to expand. ``None``
            (default) expands all levels until no struct columns remain.
            ``0`` expands no struct column.
        expand_lists: If ``True``, ``List`` and ``Array`` columns are expanded
            into one column per index position. If ``False`` (default), list
            and array columns are left as-is.

    Returns:
        A DataFrame where struct columns are recursively unnested and, if
        ``expand_lists`` is ``True``, list and array columns are expanded into
        positional columns.

    Raises:
        TypeError: If ``frame`` is not a ``polars.DataFrame``, for
            example a ``LazyFrame`` (call ``collect()`` first: a lazy
            frame has no cheap schema, and the number of rows of an
            exploded list is only known after the evaluation).
        ValueError: If ``depth`` is negative.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.nested import flatten_frame
        >>> frame = pl.DataFrame(
        ...     {
        ...         "id": [1, 2],
        ...         "coords": [
        ...             {"x": 10, "meta": {"label": "a"}},
        ...             {"x": 30, "meta": {"label": "b"}},
        ...         ],
        ...         "tags": [["foo", "bar"], ["baz"]],
        ...     }
        ... )
        >>> frame
        shape: (2, 3)
        ┌─────┬────────────┬────────────────┐
        │ id  ┆ coords     ┆ tags           │
        │ --- ┆ ---        ┆ ---            │
        │ i64 ┆ struct[2]  ┆ list[str]      │
        ╞═════╪════════════╪════════════════╡
        │ 1   ┆ {10,{"a"}} ┆ ["foo", "bar"] │
        │ 2   ┆ {30,{"b"}} ┆ ["baz"]        │
        └─────┴────────────┴────────────────┘
        >>> flatten_frame(frame)
        shape: (2, 4)
        ┌─────┬──────────┬───────────────────┬────────────────┐
        │ id  ┆ coords.x ┆ coords.meta.label ┆ tags           │
        │ --- ┆ ---      ┆ ---               ┆ ---            │
        │ i64 ┆ i64      ┆ str               ┆ list[str]      │
        ╞═════╪══════════╪═══════════════════╪════════════════╡
        │ 1   ┆ 10       ┆ a                 ┆ ["foo", "bar"] │
        │ 2   ┆ 30       ┆ b                 ┆ ["baz"]        │
        └─────┴──────────┴───────────────────┴────────────────┘
        >>> flatten_frame(frame, expand_lists=True)
        shape: (2, 5)
        ┌─────┬──────────┬───────────────────┬────────┬────────┐
        │ id  ┆ coords.x ┆ coords.meta.label ┆ tags.0 ┆ tags.1 │
        │ --- ┆ ---      ┆ ---               ┆ ---    ┆ ---    │
        │ i64 ┆ i64      ┆ str               ┆ str    ┆ str    │
        ╞═════╪══════════╪═══════════════════╪════════╪════════╡
        │ 1   ┆ 10       ┆ a                 ┆ foo    ┆ bar    │
        │ 2   ┆ 30       ┆ b                 ┆ baz    ┆ null   │
        └─────┴──────────┴───────────────────┴────────┴────────┘

        ```
    """
    if not isinstance(frame, pl.DataFrame):
        msg = (
            f"frame must be a polars DataFrame, got {type(frame).__qualname__}. "
            "Call collect() on a LazyFrame first."
        )
        raise TypeError(msg)
    if depth is not None:
        validate_non_negative(depth, name="depth")

    current_depth = 0
    while depth is None or current_depth < depth:
        has_structs = any(isinstance(dtype, pl.Struct) for dtype in frame.schema.values())
        has_nested_structs = any(_is_nested_struct(dtype) for dtype in frame.schema.values())
        if not has_structs and not has_nested_structs:
            break
        frame = _unnest_one_level(frame, separator)
        current_depth += 1

    if expand_lists:
        frame = _expand_list_columns(frame, separator)

    return frame
