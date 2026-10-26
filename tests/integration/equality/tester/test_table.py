from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola.equality import assert_objects_equal, compare, objects_are_allclose, objects_are_equal
from coola.testing.fixtures import pandas_available, polars_available, pyarrow_available
from coola.utils.imports import is_pandas_available, is_polars_available, is_pyarrow_available

if is_pandas_available():
    import pandas as pd
else:
    pd = Mock()
if is_polars_available():
    import polars as pl
else:
    pl = Mock()
if is_pyarrow_available():
    import pyarrow as pa
else:
    pa = Mock()


@pandas_available
def test_a_pandas_frame_in_a_nested_structure_is_compared_by_the_default_registry() -> None:
    """Test that a frame nested in containers is compared without any
    patching."""
    data = {"prices": [pd.DataFrame({"a": [1.0, 2.0]})]}
    assert objects_are_allclose(data, {"prices": [pd.DataFrame({"a": [1.0, 2.05]})]}, atol=0.1)


@pandas_available
def test_the_location_of_a_pandas_frame_is_the_path_to_the_frame() -> None:
    """Test that the path stops at the frame, which is one leaf."""
    result = compare({"prices": [pd.DataFrame({"a": [1]})]}, {"prices": [pd.DataFrame({"a": [2]})]})
    assert result.path == ("prices", 0)


@polars_available
def test_the_location_of_a_polars_frame_is_the_path_to_the_frame() -> None:
    result = compare({"prices": [pl.DataFrame({"a": [1]})]}, {"prices": [pl.DataFrame({"a": [2]})]})
    assert result.path == ("prices", 0)


@pyarrow_available
def test_the_location_of_an_arrow_table_is_the_path_to_the_table() -> None:
    result = compare({"prices": [pa.table({"a": [1]})]}, {"prices": [pa.table({"a": [2]})]})
    assert result.path == ("prices", 0)


@pandas_available
def test_assert_objects_equal_explains_a_pandas_frame() -> None:
    with pytest.raises(AssertionError, match=r"tables have different values: column 'a' differs"):
        assert_objects_equal(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]}))


@pandas_available
@polars_available
@pyarrow_available
def test_the_libraries_agree_on_the_same_numbers() -> None:
    """Test that the three libraries give the same answer for the same
    values and the same tolerance."""
    values, other = [1.0, 2.0, 3.0], [1.0, 2.0, 3.04]
    answers = {
        objects_are_allclose(pd.Series(values), pd.Series(other), atol=0.05),
        objects_are_allclose(pl.Series(values), pl.Series(other), atol=0.05),
        objects_are_allclose(pa.array(values), pa.array(other), atol=0.05),
    }
    assert answers == {True}


@pandas_available
@polars_available
@pyarrow_available
def test_the_libraries_agree_on_a_difference() -> None:
    answers = {
        objects_are_equal(pd.Series([1.0, 2.0]), pd.Series([1.0, 3.0])),
        objects_are_equal(pl.Series([1.0, 2.0]), pl.Series([1.0, 3.0])),
        objects_are_equal(pa.array([1.0, 2.0]), pa.array([1.0, 3.0])),
    }
    assert answers == {False}
