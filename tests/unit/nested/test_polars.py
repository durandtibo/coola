from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.nested.polars import (
    expand_list_columns,
    flatten_frame,
    is_nested_struct,
    unnest_one_level,
    unnest_with_separator,
)
from coola.testing.fixtures import polars_available
from coola.utils.imports import is_polars_available

if TYPE_CHECKING or is_polars_available():
    import polars as pl
    from polars.testing import assert_frame_equal
else:  # pragma: no cover
    pl = Mock()


######################################
#     Tests for is_nested_struct     #
######################################


@polars_available
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        pytest.param(pl.List(pl.Struct({"x": pl.Int64})), True, id="list-of-struct"),
        pytest.param(pl.Array(pl.Struct({"x": pl.Int64}), shape=2), True, id="array-of-struct"),
        pytest.param(pl.List(pl.Int64), False, id="list-of-int"),
        pytest.param(pl.Array(pl.Int64, shape=2), False, id="array-of-int"),
        pytest.param(pl.Struct({"x": pl.Int64}), False, id="struct"),
        pytest.param(pl.Int64, False, id="int"),
        pytest.param(pl.Utf8, False, id="str"),
    ],
)
def test_is_nested_struct(dtype: pl.DataType, expected: bool) -> None:
    assert is_nested_struct(dtype) == expected


###########################################
#     Tests for unnest_with_separator     #
###########################################


def unnest_without_separator(
    self: pl.DataFrame,
    columns: str | list[str],
    *more_columns: str,
    **kwargs: object,
) -> pl.DataFrame:
    """Simulate old polars behaviour by raising TypeError when separator
    is passed."""
    if "separator" in kwargs:
        msg = "DataFrame.unnest() got an unexpected keyword argument 'separator'"
        raise TypeError(msg)
    return pl.DataFrame.unnest(self, columns, *more_columns)


@polars_available
def test_unnest_with_separator_single_struct() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "coords": [{"x": 10, "y": 20}, {"x": 30, "y": 40}]},
        schema={"id": pl.Int64, "coords": pl.Struct({"x": pl.Int64, "y": pl.Int64})},
    )
    assert_frame_equal(
        unnest_with_separator(frame, ["coords"], separator="."),
        pl.DataFrame(
            {"id": [1, 2], "coords.x": [10, 30], "coords.y": [20, 40]},
            schema={"id": pl.Int64, "coords.x": pl.Int64, "coords.y": pl.Int64},
        ),
    )


@polars_available
def test_unnest_with_separator_multiple_structs() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "coords": [{"x": 10, "y": 20}, {"x": 30, "y": 40}],
            "meta": [{"label": "a"}, {"label": "b"}],
        },
        schema={
            "id": pl.Int64,
            "coords": pl.Struct({"x": pl.Int64, "y": pl.Int64}),
            "meta": pl.Struct({"label": pl.String}),
        },
    )
    assert_frame_equal(
        unnest_with_separator(frame, ["coords", "meta"], separator="."),
        pl.DataFrame(
            {"id": [1, 2], "coords.x": [10, 30], "coords.y": [20, 40], "meta.label": ["a", "b"]},
            schema={
                "id": pl.Int64,
                "coords.x": pl.Int64,
                "coords.y": pl.Int64,
                "meta.label": pl.String,
            },
        ),
    )


@polars_available
@pytest.mark.parametrize(
    "separator", [".", "_", "__"], ids=["dot", "underscore", "double-underscore"]
)
def test_unnest_with_separator_separators(separator: str) -> None:
    frame = pl.DataFrame(
        {"coords": [{"x": 1, "y": 2}]},
        schema={"coords": pl.Struct({"x": pl.Int64, "y": pl.Int64})},
    )
    assert_frame_equal(
        unnest_with_separator(frame, ["coords"], separator=separator),
        pl.DataFrame(
            {f"coords{separator}x": [1], f"coords{separator}y": [2]},
            schema={f"coords{separator}x": pl.Int64, f"coords{separator}y": pl.Int64},
        ),
    )


@polars_available
def test_unnest_with_separator_fallback_single_struct() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "coords": [{"x": 10, "y": 20}, {"x": 30, "y": 40}]},
        schema={"id": pl.Int64, "coords": pl.Struct({"x": pl.Int64, "y": pl.Int64})},
    )
    with patch.object(pl.DataFrame, "unnest", unnest_without_separator):
        assert_frame_equal(
            unnest_with_separator(frame, ["coords"], separator="."),
            pl.DataFrame(
                {"id": [1, 2], "coords.x": [10, 30], "coords.y": [20, 40]},
                schema={"id": pl.Int64, "coords.x": pl.Int64, "coords.y": pl.Int64},
            ),
        )


@polars_available
def test_unnest_with_separator_fallback_multiple_structs() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "coords": [{"x": 10, "y": 20}, {"x": 30, "y": 40}],
            "meta": [{"label": "a"}, {"label": "b"}],
        },
        schema={
            "id": pl.Int64,
            "coords": pl.Struct({"x": pl.Int64, "y": pl.Int64}),
            "meta": pl.Struct({"label": pl.String}),
        },
    )
    with patch.object(pl.DataFrame, "unnest", unnest_without_separator):
        assert_frame_equal(
            unnest_with_separator(frame, ["coords", "meta"], separator="."),
            pl.DataFrame(
                {
                    "id": [1, 2],
                    "coords.x": [10, 30],
                    "coords.y": [20, 40],
                    "meta.label": ["a", "b"],
                },
                schema={
                    "id": pl.Int64,
                    "coords.x": pl.Int64,
                    "coords.y": pl.Int64,
                    "meta.label": pl.String,
                },
            ),
        )


@polars_available
@pytest.mark.parametrize(
    "separator", [".", "_", "__"], ids=["dot", "underscore", "double-underscore"]
)
def test_unnest_with_separator_fallback_separators(separator: str) -> None:
    frame = pl.DataFrame(
        {"coords": [{"x": 1, "y": 2}]},
        schema={"coords": pl.Struct({"x": pl.Int64, "y": pl.Int64})},
    )
    with patch.object(pl.DataFrame, "unnest", unnest_without_separator):
        assert_frame_equal(
            unnest_with_separator(frame, ["coords"], separator=separator),
            pl.DataFrame(
                {f"coords{separator}x": [1], f"coords{separator}y": [2]},
                schema={f"coords{separator}x": pl.Int64, f"coords{separator}y": pl.Int64},
            ),
        )


######################################
#     Tests for unnest_one_level     #
######################################


@polars_available
def test_unnest_one_level_flat_passthrough() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "value": ["a", "b"]},
        schema={"id": pl.Int64, "value": pl.String},
    )
    assert_frame_equal(
        unnest_one_level(frame, separator="."),
        pl.DataFrame(
            {"id": [1, 2], "value": ["a", "b"]},
            schema={"id": pl.Int64, "value": pl.String},
        ),
    )


@polars_available
def test_unnest_one_level_single_struct() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "coords": [{"x": 10, "y": 20}, {"x": 30, "y": 40}]},
        schema={"id": pl.Int64, "coords": pl.Struct({"x": pl.Int64, "y": pl.Int64})},
    )
    assert_frame_equal(
        unnest_one_level(frame, separator="."),
        pl.DataFrame(
            {"id": [1, 2], "coords.x": [10, 30], "coords.y": [20, 40]},
            schema={"id": pl.Int64, "coords.x": pl.Int64, "coords.y": pl.Int64},
        ),
    )


@polars_available
def test_unnest_one_level_list_of_struct() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "items": [[{"name": "a"}], [{"name": "b"}, {"name": "c"}]]},
        schema={"id": pl.Int64, "items": pl.List(pl.Struct({"name": pl.String}))},
    )
    assert_frame_equal(
        unnest_one_level(frame, separator="."),
        pl.DataFrame(
            {"id": [1, 2, 2], "items.name": ["a", "b", "c"]},
            schema={"id": pl.Int64, "items.name": pl.String},
        ),
    )


#####################################################
#     Tests for expand_list_columns — DataFrame     #
#####################################################


@polars_available
def test_expand_list_columns_no_lists() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "value": ["a", "b"]},
        schema={"id": pl.Int64, "value": pl.String},
    )
    assert_frame_equal(
        expand_list_columns(frame, separator="."),
        pl.DataFrame(
            {"id": [1, 2], "value": ["a", "b"]},
            schema={"id": pl.Int64, "value": pl.String},
        ),
    )


@polars_available
def test_expand_list_columns_equal_length_lists() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "tags": [["a", "b"], ["c", "d"]]},
        schema={"id": pl.Int64, "tags": pl.List(pl.String)},
    )
    assert_frame_equal(
        expand_list_columns(frame, separator="."),
        pl.DataFrame(
            {"id": [1, 2], "tags.0": ["a", "c"], "tags.1": ["b", "d"]},
            schema={"id": pl.Int64, "tags.0": pl.String, "tags.1": pl.String},
        ),
    )


@polars_available
def test_expand_list_columns_unequal_length_lists() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "tags": [["foo", "bar"], ["baz"]]},
        schema={"id": pl.Int64, "tags": pl.List(pl.String)},
    )
    assert_frame_equal(
        expand_list_columns(frame, separator="."),
        pl.DataFrame(
            {"id": [1, 2], "tags.0": ["foo", "baz"], "tags.1": ["bar", None]},
            schema={"id": pl.Int64, "tags.0": pl.String, "tags.1": pl.String},
        ),
    )


@polars_available
def test_expand_list_columns_single_element_lists() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "tags": [["a"], ["b"]]},
        schema={"id": pl.Int64, "tags": pl.List(pl.String)},
    )
    assert_frame_equal(
        expand_list_columns(frame, separator="."),
        pl.DataFrame(
            {"id": [1, 2], "tags.0": ["a", "b"]},
            schema={"id": pl.Int64, "tags.0": pl.String},
        ),
    )


@polars_available
def test_expand_list_columns_empty_list() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "tags": [["a"], []]},
        schema={"id": pl.Int64, "tags": pl.List(pl.String)},
    )
    assert_frame_equal(
        expand_list_columns(frame, separator="."),
        pl.DataFrame(
            {"id": [1, 2], "tags.0": ["a", None]},
            schema={"id": pl.Int64, "tags.0": pl.String},
        ),
    )


@polars_available
def test_expand_list_columns_drops_original() -> None:
    frame = pl.DataFrame(
        {"tags": [["a", "b"], ["c"]]},
        schema={"tags": pl.List(pl.String)},
    )
    result = expand_list_columns(frame, separator=".")
    assert "tags" not in result.columns


@polars_available
def test_expand_list_columns_multiple_list_columns() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "tags": [["a", "b"], ["c"]], "scores": [[1.0, 2.0, 3.0], [4.0]]},
        schema={"id": pl.Int64, "tags": pl.List(pl.String), "scores": pl.List(pl.Float64)},
    )
    assert_frame_equal(
        expand_list_columns(frame, separator="."),
        pl.DataFrame(
            {
                "id": [1, 2],
                "tags.0": ["a", "c"],
                "tags.1": ["b", None],
                "scores.0": [1.0, 4.0],
                "scores.1": [2.0, None],
                "scores.2": [3.0, None],
            },
            schema={
                "id": pl.Int64,
                "tags.0": pl.String,
                "tags.1": pl.String,
                "scores.0": pl.Float64,
                "scores.1": pl.Float64,
                "scores.2": pl.Float64,
            },
        ),
    )


@polars_available
@pytest.mark.parametrize(
    "separator", [".", "_", "__"], ids=["dot", "underscore", "double-underscore"]
)
def test_expand_list_columns_separator(separator: str) -> None:
    frame = pl.DataFrame(
        {"tags": [["a", "b"], ["c"]]},
        schema={"tags": pl.List(pl.String)},
    )
    assert_frame_equal(
        expand_list_columns(frame, separator=separator),
        pl.DataFrame(
            {f"tags{separator}0": ["a", "c"], f"tags{separator}1": ["b", None]},
            schema={f"tags{separator}0": pl.String, f"tags{separator}1": pl.String},
        ),
    )


###################################
#     Tests for flatten_frame     #
###################################

# ---------------------------------------------------------------------------
# flatten_frame — validation
# ---------------------------------------------------------------------------


@polars_available
def test_flatten_frame_invalid_depth() -> None:
    frame = pl.DataFrame({"id": [1]}, schema={"id": pl.Int64})
    with pytest.raises(ValueError, match="depth"):
        flatten_frame(frame, depth=0)


@polars_available
def test_flatten_frame_negative_depth() -> None:
    frame = pl.DataFrame({"id": [1]}, schema={"id": pl.Int64})
    with pytest.raises(ValueError, match="depth"):
        flatten_frame(frame, depth=-1)


# ---------------------------------------------------------------------------
# flatten_frame — no-op cases
# ---------------------------------------------------------------------------


@polars_available
def test_flatten_frame_empty() -> None:
    frame = pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)}, schema={"id": pl.Int64})
    assert_frame_equal(
        flatten_frame(frame),
        pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)}, schema={"id": pl.Int64}),
    )


@polars_available
def test_flatten_frame_no_nested_columns() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "value": ["a", "b"]},
        schema={"id": pl.Int64, "value": pl.String},
    )
    assert_frame_equal(
        flatten_frame(frame),
        pl.DataFrame(
            {"id": [1, 2], "value": ["a", "b"]},
            schema={"id": pl.Int64, "value": pl.String},
        ),
    )


# ---------------------------------------------------------------------------
# flatten_frame — struct unnesting
# ---------------------------------------------------------------------------


@polars_available
def test_flatten_frame_single_struct() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "coords": [{"x": 10, "y": 20}, {"x": 30, "y": 40}]},
        schema={"id": pl.Int64, "coords": pl.Struct({"x": pl.Int64, "y": pl.Int64})},
    )
    assert_frame_equal(
        flatten_frame(frame),
        pl.DataFrame(
            {"id": [1, 2], "coords.x": [10, 30], "coords.y": [20, 40]},
            schema={"id": pl.Int64, "coords.x": pl.Int64, "coords.y": pl.Int64},
        ),
    )


@polars_available
def test_flatten_frame_two_level_nested_struct() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "coords": [{"x": 10, "meta": {"label": "a"}}, {"x": 30, "meta": {"label": "b"}}],
        },
        schema={
            "id": pl.Int64,
            "coords": pl.Struct({"x": pl.Int64, "meta": pl.Struct({"label": pl.String})}),
        },
    )
    assert_frame_equal(
        flatten_frame(frame),
        pl.DataFrame(
            {"id": [1, 2], "coords.x": [10, 30], "coords.meta.label": ["a", "b"]},
            schema={"id": pl.Int64, "coords.x": pl.Int64, "coords.meta.label": pl.String},
        ),
    )


@polars_available
def test_flatten_frame_three_level_nested_struct() -> None:
    frame = pl.DataFrame(
        {"id": [1], "a": [{"b": {"c": {"d": 42}}}]},
        schema={
            "id": pl.Int64,
            "a": pl.Struct({"b": pl.Struct({"c": pl.Struct({"d": pl.Int64})})}),
        },
    )
    assert_frame_equal(
        flatten_frame(frame),
        pl.DataFrame(
            {"id": [1], "a.b.c.d": [42]},
            schema={"id": pl.Int64, "a.b.c.d": pl.Int64},
        ),
    )


@polars_available
@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        pytest.param(
            1,
            pl.DataFrame(
                {"id": [1], "a.b": [{"c": {"d": 42}}]},
                schema={"id": pl.Int64, "a.b": pl.Struct({"c": pl.Struct({"d": pl.Int64})})},
            ),
            id="depth-1",
        ),
        pytest.param(
            2,
            pl.DataFrame(
                {"id": [1], "a.b.c": [{"d": 42}]},
                schema={"id": pl.Int64, "a.b.c": pl.Struct({"d": pl.Int64})},
            ),
            id="depth-2",
        ),
        pytest.param(
            3,
            pl.DataFrame(
                {"id": [1], "a.b.c.d": [42]},
                schema={"id": pl.Int64, "a.b.c.d": pl.Int64},
            ),
            id="depth-3",
        ),
    ],
)
def test_flatten_frame_depth_limit(depth: int, expected: pl.DataFrame) -> None:
    frame = pl.DataFrame(
        {"id": [1], "a": [{"b": {"c": {"d": 42}}}]},
        schema={
            "id": pl.Int64,
            "a": pl.Struct({"b": pl.Struct({"c": pl.Struct({"d": pl.Int64})})}),
        },
    )
    assert_frame_equal(flatten_frame(frame, depth=depth), expected)


@polars_available
@pytest.mark.parametrize(
    "separator", [".", "_", "__"], ids=["dot", "underscore", "double-underscore"]
)
def test_flatten_frame_separator(separator: str) -> None:
    frame = pl.DataFrame(
        {"coords": [{"x": 1, "y": 2}]},
        schema={"coords": pl.Struct({"x": pl.Int64, "y": pl.Int64})},
    )
    assert_frame_equal(
        flatten_frame(frame, separator=separator),
        pl.DataFrame(
            {f"coords{separator}x": [1], f"coords{separator}y": [2]},
            schema={f"coords{separator}x": pl.Int64, f"coords{separator}y": pl.Int64},
        ),
    )


# ---------------------------------------------------------------------------
# flatten_frame — list expanding
# ---------------------------------------------------------------------------


@polars_available
def test_flatten_frame_expand_lists() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "tags": [["foo", "bar"], ["baz"]]},
        schema={"id": pl.Int64, "tags": pl.List(pl.String)},
    )
    assert_frame_equal(
        flatten_frame(frame, expand_lists=True),
        pl.DataFrame(
            {"id": [1, 2], "tags.0": ["foo", "baz"], "tags.1": ["bar", None]},
            schema={"id": pl.Int64, "tags.0": pl.String, "tags.1": pl.String},
        ),
    )


@polars_available
def test_flatten_frame_lists_not_expanded_by_default() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "tags": [["foo", "bar"], ["baz"]]},
        schema={"id": pl.Int64, "tags": pl.List(pl.String)},
    )
    assert_frame_equal(
        flatten_frame(frame),
        pl.DataFrame(
            {"id": [1, 2], "tags": [["foo", "bar"], ["baz"]]},
            schema={"id": pl.Int64, "tags": pl.List(pl.String)},
        ),
    )


# ---------------------------------------------------------------------------
# flatten_frame — list of structs
# ---------------------------------------------------------------------------


@polars_available
def test_flatten_frame_list_of_structs() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "items": [
                [{"name": "a", "val": 1}],
                [{"name": "b", "val": 2}, {"name": "c", "val": 3}],
            ],
        },
        schema={"id": pl.Int64, "items": pl.List(pl.Struct({"name": pl.String, "val": pl.Int64}))},
    )
    assert_frame_equal(
        flatten_frame(frame),
        pl.DataFrame(
            {"id": [1, 2, 2], "items.name": ["a", "b", "c"], "items.val": [1, 2, 3]},
            schema={"id": pl.Int64, "items.name": pl.String, "items.val": pl.Int64},
        ),
    )


@polars_available
def test_flatten_frame_list_of_nested_structs() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "items": [
                [{"name": "a", "meta": {"score": 1.0}}],
                [{"name": "b", "meta": {"score": 2.0}}, {"name": "c", "meta": {"score": 3.0}}],
            ],
        },
        schema={
            "id": pl.Int64,
            "items": pl.List(
                pl.Struct({"name": pl.String, "meta": pl.Struct({"score": pl.Float64})})
            ),
        },
    )
    assert_frame_equal(
        flatten_frame(frame),
        pl.DataFrame(
            {"id": [1, 2, 2], "items.name": ["a", "b", "c"], "items.meta.score": [1.0, 2.0, 3.0]},
            schema={"id": pl.Int64, "items.name": pl.String, "items.meta.score": pl.Float64},
        ),
    )


# ---------------------------------------------------------------------------
# flatten_frame — struct containing lists
# ---------------------------------------------------------------------------


@polars_available
def test_flatten_frame_struct_with_list_field() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "data": [{"tags": ["a", "b"], "score": 1}, {"tags": ["c"], "score": 2}]},
        schema={"id": pl.Int64, "data": pl.Struct({"tags": pl.List(pl.String), "score": pl.Int64})},
    )
    assert_frame_equal(
        flatten_frame(frame),
        pl.DataFrame(
            {"id": [1, 2], "data.tags": [["a", "b"], ["c"]], "data.score": [1, 2]},
            schema={"id": pl.Int64, "data.tags": pl.List(pl.String), "data.score": pl.Int64},
        ),
    )


@polars_available
def test_flatten_frame_struct_with_list_field_expanded() -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "data": [{"tags": ["a", "b"], "score": 1}, {"tags": ["c"], "score": 2}]},
        schema={"id": pl.Int64, "data": pl.Struct({"tags": pl.List(pl.String), "score": pl.Int64})},
    )
    assert_frame_equal(
        flatten_frame(frame, expand_lists=True),
        pl.DataFrame(
            {
                "id": [1, 2],
                "data.score": [1, 2],
                "data.tags.0": ["a", "c"],
                "data.tags.1": ["b", None],
            },
            schema={
                "id": pl.Int64,
                "data.score": pl.Int64,
                "data.tags.0": pl.String,
                "data.tags.1": pl.String,
            },
        ),
    )
