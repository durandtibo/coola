from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from unittest.mock import Mock, patch

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators import (
    BaseEqualityComparator,
    DefaultEqualityComparator,
    JaxArrayEqualityComparator,
    MappingEqualityComparator,
    NumpyArrayEqualityComparator,
    NumpyMaskedArrayEqualityComparator,
    PandasDataFrameEqualityComparator,
    PandasSeriesEqualityComparator,
    PolarsDataFrameEqualityComparator,
    PolarsSeriesEqualityComparator,
    SequenceEqualityComparator,
    TorchPackedSequenceEqualityComparator,
    TorchTensorEqualityComparator,
    XarrayDataArrayEqualityComparator,
    XarrayDatasetEqualityComparator,
    XarrayVariableEqualityComparator,
)
from coola.equality.testers import EqualityTester, LocalEqualityTester
from coola.testing import (
    jax_available,
    numpy_available,
    pandas_available,
    polars_available,
    torch_available,
    xarray_available,
)
from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
    is_xarray_available,
)

if is_jax_available():
    import jax.numpy as jnp
if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_torch_available():
    import torch
if is_xarray_available():
    import xarray as xr


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


####################################
#     Tests for EqualityTester     #
####################################


def test_equality_tester_str() -> None:
    assert str(EqualityTester()).startswith("EqualityTester(")


@patch.dict(EqualityTester.registry, {}, clear=True)
def test_equality_tester_add_comparator() -> None:
    tester = EqualityTester()
    comparator = Mock(spec=BaseEqualityComparator)
    tester.add_comparator(int, comparator)
    assert tester.registry[int] == comparator


@patch.dict(EqualityTester.registry, {}, clear=True)
def test_equality_tester_add_comparator_duplicate_exist_ok_true() -> None:
    tester = EqualityTester()
    comparator = Mock(spec=BaseEqualityComparator)
    tester.add_comparator(int, Mock(spec=BaseEqualityComparator))
    tester.add_comparator(int, comparator, exist_ok=True)
    assert tester.registry[int] == comparator


@patch.dict(EqualityTester.registry, {}, clear=True)
def test_equality_tester_add_comparator_duplicate_exist_ok_false() -> None:
    tester = EqualityTester()
    comparator = Mock(spec=BaseEqualityComparator)
    tester.add_comparator(int, Mock(spec=BaseEqualityComparator))
    with pytest.raises(RuntimeError, match="An comparator (.*) is already registered"):
        tester.add_comparator(int, comparator)


def test_equality_tester_equal_true(config: EqualityConfig) -> None:
    assert EqualityTester().equal(actual=1, expected=1, config=config)


def test_equality_tester_equal_false(config: EqualityConfig) -> None:
    assert not EqualityTester().equal(actual=1, expected=2, config=config)


def test_equality_tester_has_comparator_true() -> None:
    assert EqualityTester().has_comparator(dict)


def test_equality_tester_has_comparator_false() -> None:
    assert not EqualityTester().has_comparator(type(None))


def test_equality_tester_find_comparator_direct() -> None:
    assert isinstance(EqualityTester().find_comparator(dict), MappingEqualityComparator)


def test_equality_tester_find_comparator_indirect() -> None:
    assert isinstance(EqualityTester().find_comparator(str), DefaultEqualityComparator)


def test_equality_tester_find_comparator_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect data type:"):
        EqualityTester().find_comparator(Mock(__mro__=[]))


@patch.dict(EqualityTester.registry, {object: DefaultEqualityComparator()}, clear=True)
def test_equality_tester_local_copy() -> None:
    tester = EqualityTester.local_copy()
    tester.add_comparator(dict, MappingEqualityComparator())
    assert EqualityTester.registry == {object: DefaultEqualityComparator()}
    assert tester == LocalEqualityTester(
        {dict: MappingEqualityComparator(), object: DefaultEqualityComparator()}
    )


def test_equality_tester_registry_default() -> None:
    assert len(EqualityTester.registry) >= 7
    assert isinstance(EqualityTester.registry[Mapping], MappingEqualityComparator)
    assert isinstance(EqualityTester.registry[Sequence], SequenceEqualityComparator)
    assert isinstance(EqualityTester.registry[deque], SequenceEqualityComparator)
    assert isinstance(EqualityTester.registry[dict], MappingEqualityComparator)
    assert isinstance(EqualityTester.registry[list], SequenceEqualityComparator)
    assert isinstance(EqualityTester.registry[object], DefaultEqualityComparator)
    assert isinstance(EqualityTester.registry[tuple], SequenceEqualityComparator)


@jax_available
def test_equality_tester_registry_jax() -> None:
    assert isinstance(EqualityTester.registry[jnp.ndarray], JaxArrayEqualityComparator)


@numpy_available
def test_equality_tester_registry_numpy() -> None:
    assert isinstance(EqualityTester.registry[np.ndarray], NumpyArrayEqualityComparator)
    assert isinstance(
        EqualityTester.registry[np.ma.MaskedArray], NumpyMaskedArrayEqualityComparator
    )


@pandas_available
def test_equality_tester_registry_pandas() -> None:
    assert isinstance(EqualityTester.registry[pd.DataFrame], PandasDataFrameEqualityComparator)
    assert isinstance(EqualityTester.registry[pd.Series], PandasSeriesEqualityComparator)


@polars_available
def test_equality_tester_registry_polars() -> None:
    assert isinstance(EqualityTester.registry[pl.DataFrame], PolarsDataFrameEqualityComparator)
    assert isinstance(EqualityTester.registry[pl.Series], PolarsSeriesEqualityComparator)


@torch_available
def test_equality_tester_registry_torch() -> None:
    assert isinstance(EqualityTester.registry[torch.Tensor], TorchTensorEqualityComparator)
    assert isinstance(
        EqualityTester.registry[torch.nn.utils.rnn.PackedSequence],
        TorchPackedSequenceEqualityComparator,
    )


@xarray_available
def test_equality_tester_registry_xarray() -> None:
    assert isinstance(EqualityTester.registry[xr.DataArray], XarrayDataArrayEqualityComparator)
    assert isinstance(EqualityTester.registry[xr.Dataset], XarrayDatasetEqualityComparator)
    assert isinstance(EqualityTester.registry[xr.Variable], XarrayVariableEqualityComparator)


#########################################
#     Tests for LocalEqualityTester     #
#########################################


def test_local_equality_tester_str() -> None:
    assert str(LocalEqualityTester()).startswith("LocalEqualityTester(")


def test_local_equality_tester__eq__true() -> None:
    assert LocalEqualityTester({object: DefaultEqualityComparator()}) == LocalEqualityTester(
        {object: DefaultEqualityComparator()}
    )


def test_local_equality_tester__eq__true_empty() -> None:
    assert LocalEqualityTester(None) == LocalEqualityTester({})


def test_local_equality_tester__eq__false_different_key() -> None:
    assert LocalEqualityTester({object: DefaultEqualityComparator()}) != LocalEqualityTester(
        {int: DefaultEqualityComparator()}
    )


def test_local_equality_tester__eq__false_different_value() -> None:
    assert LocalEqualityTester({object: DefaultEqualityComparator()}) != LocalEqualityTester(
        {object: MappingEqualityComparator()}
    )


def test_local_equality_tester__eq__false_different_type() -> None:
    assert LocalEqualityTester() != 1


def test_local_equality_tester_hash() -> None:
    assert isinstance(hash(LocalEqualityTester({})), int)


def test_local_equality_tester_hash_same() -> None:
    assert hash(LocalEqualityTester({})) == hash(LocalEqualityTester({}))


def test_local_equality_tester_hash_different() -> None:
    assert hash(LocalEqualityTester({})) != hash(
        LocalEqualityTester({int: DefaultEqualityComparator()})
    )


def test_local_equality_tester_registry_default() -> None:
    assert LocalEqualityTester().registry == {}


def test_local_equality_tester_add_comparator() -> None:
    tester = LocalEqualityTester()
    comparator = Mock(spec=BaseEqualityComparator)
    tester.add_comparator(int, comparator)
    assert tester == LocalEqualityTester({int: comparator})


def test_local_equality_tester_add_comparator_duplicate_exist_ok_true() -> None:
    tester = LocalEqualityTester()
    comparator = Mock(spec=BaseEqualityComparator)
    tester.add_comparator(int, Mock(spec=BaseEqualityComparator))
    tester.add_comparator(int, comparator, exist_ok=True)
    assert tester == LocalEqualityTester({int: comparator})


def test_local_equality_tester_add_comparator_duplicate_exist_ok_false() -> None:
    tester = LocalEqualityTester()
    comparator = Mock(spec=BaseEqualityComparator)
    tester.add_comparator(int, Mock(spec=BaseEqualityComparator))
    with pytest.raises(RuntimeError, match="An comparator (.*) is already registered"):
        tester.add_comparator(int, comparator)


def test_local_equality_tester_clone() -> None:
    tester = LocalEqualityTester({dict: MappingEqualityComparator()})
    tester_cloned = tester.clone()
    tester.add_comparator(list, SequenceEqualityComparator())
    tester_cloned.add_comparator(object, DefaultEqualityComparator())
    assert tester == LocalEqualityTester(
        {dict: MappingEqualityComparator(), list: SequenceEqualityComparator()}
    )
    assert tester_cloned == LocalEqualityTester(
        {
            dict: MappingEqualityComparator(),
            object: DefaultEqualityComparator(),
        }
    )


def test_local_equality_tester_equal_true(config: EqualityConfig) -> None:
    assert LocalEqualityTester({object: DefaultEqualityComparator()}).equal(
        actual=1, expected=1, config=config
    )


def test_local_equality_tester_equal_false(config: EqualityConfig) -> None:
    assert not LocalEqualityTester({object: DefaultEqualityComparator()}).equal(
        actual=1, expected=2, config=config
    )


def test_local_equality_tester_has_comparator_true() -> None:
    assert LocalEqualityTester({dict: MappingEqualityComparator()}).has_comparator(dict)


def test_local_equality_tester_has_comparator_false() -> None:
    assert not LocalEqualityTester().has_comparator(type(None))


def test_local_equality_tester_find_comparator_direct() -> None:
    assert isinstance(
        LocalEqualityTester({dict: MappingEqualityComparator()}).find_comparator(dict),
        MappingEqualityComparator,
    )


def test_local_equality_tester_find_comparator_indirect() -> None:
    assert isinstance(
        LocalEqualityTester(
            {dict: MappingEqualityComparator(), object: DefaultEqualityComparator()}
        ).find_comparator(str),
        DefaultEqualityComparator,
    )


def test_local_equality_tester_find_comparator_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Incorrect data type:"):
        LocalEqualityTester().find_comparator(Mock(__mro__=[]))
