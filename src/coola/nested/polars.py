r"""Contain functions for polars DataFrame/LazyFrame."""

from __future__ import annotations

__all__ = ["expand_list_columns", "flatten_frame", "is_nested_struct", "unnest_one_level"]

from typing import TYPE_CHECKING

from coola.utils.imports import is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
else:  # pragma: no cover
    from coola.utils.fallback.polars import polars as pl


def is_nested_struct(dtype: pl.DataType) -> bool:
    """Return True if dtype is a List or Array whose inner type is a
    Struct."""
    if isinstance(dtype, (pl.List, pl.Array)):
        return isinstance(dtype.inner, pl.Struct)
    return False


def expand_list_columns(
    frame: pl.DataFrame | pl.LazyFrame,
    separator: str,
) -> pl.DataFrame | pl.LazyFrame:
    """Expand list and array columns into one column per index position.

    For each ``List`` or ``Array`` column, creates new columns named
    ``<col><separator><index>`` for each position up to the maximum list
    length across all rows. Rows with lists shorter than the maximum are
    filled with ``None``. The original list columns are dropped.

    Args:
        frame: A DataFrame or LazyFrame that may contain list or array columns.
        separator: Separator used to build output column names following the
            ``<col><separator><index>`` pattern.

    Returns:
        A DataFrame or LazyFrame with list and array columns replaced by
        their expanded positional columns. Non-list columns are unchanged.
        If no list or array columns are present the frame is returned as-is.
    """
    list_cols = [
        name for name, dtype in frame.schema.items() if isinstance(dtype, (pl.List, pl.Array))
    ]
    if not list_cols:
        return frame

    # Collect to DataFrame to compute max lengths — required for both input types.
    collected = frame.collect() if isinstance(frame, pl.LazyFrame) else frame

    expansions: dict[str, pl.Series] = {}
    for col in list_cols:
        max_len = collected[col].list.len().max() or 0
        for i in range(max_len):
            expansions[f"{col}{separator}{i}"] = collected[col].list.get(i, null_on_oob=True)

    result = collected.drop(list_cols).with_columns(
        pl.Series(name, values) for name, values in expansions.items()
    )
    return result.lazy() if isinstance(frame, pl.LazyFrame) else result


def unnest_one_level(
    frame: pl.DataFrame | pl.LazyFrame,
    separator: str,
) -> pl.DataFrame | pl.LazyFrame:
    """Unnest all top-level struct columns in a DataFrame or LazyFrame
    by one level.

    Columns that are lists or arrays of structs are exploded first to expose
    the inner structs, which are then unnested.

    Args:
        frame: A DataFrame or LazyFrame that may contain struct columns.
        separator: Separator used to build output column names following the
            ``<parent><separator><field>`` pattern.

    Returns:
        A DataFrame or LazyFrame with all top-level struct columns unnested
        by one level. Non-struct columns are passed through unchanged. If no
        struct columns are present the frame is returned as-is.
    """
    # Explode list[struct] and array[struct] columns first to expose inner structs.
    nested_struct_cols = [name for name, dtype in frame.schema.items() if is_nested_struct(dtype)]
    if nested_struct_cols:
        frame = frame.explode(nested_struct_cols)

    struct_cols = [name for name, dtype in frame.schema.items() if isinstance(dtype, pl.Struct)]
    if not struct_cols:
        return frame
    if isinstance(frame, pl.LazyFrame):
        return frame.with_columns(
            pl.col(col).struct.unnest().name.prefix(f"{col}{separator}") for col in struct_cols
        ).drop(struct_cols)
    return frame.unnest(struct_cols, separator=separator)


def flatten_frame(
    frame: pl.DataFrame | pl.LazyFrame,
    separator: str = ".",
    depth: int | None = None,
    expand_lists: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    r"""Recursively unnest struct columns and optionally expand
    list/array columns.

    Works with both ``pl.DataFrame`` and ``pl.LazyFrame``. Non-nested columns
    are passed through unchanged. Struct columns are expanded level by level up
    to ``depth`` levels. Columns that are lists or arrays of structs are
    automatically exploded to expose their inner structs before unnesting.
    If ``expand_lists`` is enabled, remaining ``List`` and ``Array`` columns
    are expanded into one column per index position following the
    ``<col><separator><index>`` pattern, filled with ``None`` where the list
    is shorter than the maximum length.

    Args:
        frame: A DataFrame or LazyFrame that may contain struct, list, or array
            columns at any nesting level.
        separator: Separator used to build output column names following the
            ``<parent><separator><field>`` pattern. Defaults to ``"."``.
        depth: Maximum number of struct nesting levels to expand. ``None``
            (default) expands all levels until no struct columns remain.
        expand_lists: If ``True``, ``List`` and ``Array`` columns are expanded
            into one column per index position. If ``False`` (default), list
            and array columns are left as-is.

    Returns:
        A DataFrame or LazyFrame (matching the input type) where struct columns
        are recursively unnested and, if ``expand_lists`` is ``True``, list and
        array columns are expanded into positional columns.

    Raises:
        ValueError: If ``depth`` is not a positive integer or ``None``.

    Example:
        ```pycon
        >>> import polars as pl
        >>> from coola.nested.polars import flatten_frame
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
    if depth is not None and depth < 1:
        msg = f"depth must be a positive integer or None, got {depth!r}"
        raise ValueError(msg)

    current_depth = 0
    while depth is None or current_depth < depth:
        has_structs = any(isinstance(dtype, pl.Struct) for dtype in frame.schema.values())
        has_nested_structs = any(is_nested_struct(dtype) for dtype in frame.schema.values())
        if not has_structs and not has_nested_structs:
            break
        frame = unnest_one_level(frame, separator)
        current_depth += 1

    if expand_lists:
        frame = expand_list_columns(frame, separator)

    return frame
