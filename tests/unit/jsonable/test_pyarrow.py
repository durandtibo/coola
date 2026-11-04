from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.jsonable import to_jsonable
from coola.jsonable.pyarrow import get_default_transformers
from coola.testing.marks import pyarrow_available
from coola.utils.imports import is_pyarrow_available

if is_pyarrow_available():
    import pyarrow as pa


@pyarrow_available
def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [pa.Array, pa.ChunkedArray]


def test_get_default_transformers_without_pyarrow() -> None:
    with patch("coola.jsonable.pyarrow.is_pyarrow_available", lambda: False):
        assert get_default_transformers() == {}


@pyarrow_available
def test_to_jsonable_array() -> None:
    assert to_jsonable(pa.array([1, 2])) == [1, 2]


@pyarrow_available
def test_to_jsonable_array_with_a_null() -> None:
    assert to_jsonable(pa.array([1.5, None])) == [1.5, None]


@pyarrow_available
def test_to_jsonable_chunked_array() -> None:
    assert to_jsonable(pa.chunked_array([[1], [2]])) == [1, 2]


@pyarrow_available
def test_to_jsonable_array_of_strings() -> None:
    assert to_jsonable(pa.array(["a", "b"])) == ["a", "b"]


@pyarrow_available
def test_to_jsonable_array_in_a_container() -> None:
    assert to_jsonable({"a": pa.array([1])}) == {"a": [1]}


@pyarrow_available
def test_to_jsonable_table_is_not_supported() -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type Table"):
        to_jsonable(pa.table({"a": [1]}))
