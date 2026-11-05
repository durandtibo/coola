from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from coola.hashing import hash_object
from coola.hashing.pandas import _normalize_missing
from coola.testing.marks import pandas_available
from coola.utils.imports import is_pandas_available

if is_pandas_available():
    import pandas as pd
else:
    pd = Mock()

##########################################
#     Tests for missing values in object #
##########################################


@pandas_available
@pytest.mark.parametrize("missing", [np.nan, pd.NaT, pd.NA])
def test_hash_object_series_missing_values_of_an_object_column_are_the_same(
    missing: object,
) -> None:
    assert hash_object(pd.Series(["a", None], dtype=object)) == hash_object(
        pd.Series(["a", missing], dtype=object)
    )


@pandas_available
@pytest.mark.parametrize("missing", [np.nan, pd.NaT, pd.NA])
def test_hash_object_frame_missing_values_of_an_object_column_are_the_same(missing: object) -> None:
    assert hash_object(pd.DataFrame({"x": pd.Series(["a", None], dtype=object)})) == hash_object(
        pd.DataFrame({"x": pd.Series(["a", missing], dtype=object)})
    )


@pandas_available
def test_hash_object_series_missing_value_differs_from_a_string() -> None:
    assert hash_object(pd.Series(["a", None], dtype=object)) != hash_object(
        pd.Series(["a", "None"], dtype=object)
    )


@pandas_available
def test_hash_object_object_index_missing_values_are_the_same() -> None:
    assert hash_object(pd.Series([1, 2], index=pd.Index(["a", None], dtype=object))) == hash_object(
        pd.Series([1, 2], index=pd.Index(["a", np.nan], dtype=object))
    )


@pandas_available
def test_hash_object_categorical_series_different_order_of_categories() -> None:
    assert hash_object(pd.Series(pd.Categorical(["a"], categories=["a", "b"]))) != hash_object(
        pd.Series(pd.Categorical(["a"], categories=["b", "a"]))
    )


@pandas_available
def test_normalize_missing_object_array() -> None:
    values = np.array(["a", np.nan, pd.NA, pd.NaT, None, "b"], dtype=object)
    assert _normalize_missing(values).tolist() == ["a", None, None, None, None, "b"]


@pandas_available
def test_normalize_missing_does_not_modify_the_input() -> None:
    values = np.array(["a", np.nan], dtype=object)
    _normalize_missing(values)
    assert values.tolist()[0] == "a"
    assert np.isnan(values[1])


@pandas_available
def test_normalize_missing_object_array_without_missing_values_is_returned_as_is() -> None:
    values = np.array(["a", "b"], dtype=object)
    assert _normalize_missing(values) is values


@pandas_available
def test_normalize_missing_float_array_is_unchanged() -> None:
    values = np.array([1.0, np.nan])
    assert _normalize_missing(values) is values
