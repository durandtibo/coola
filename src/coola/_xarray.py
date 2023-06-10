from __future__ import annotations

import logging
from typing import Any

from coola.allclose import AllCloseTester, BaseAllCloseOperator, BaseAllCloseTester
from coola.equality import (
    BaseEqualityOperator,
    BaseEqualityTester,
    EqualityTester,
    objects_are_equal,
)
from coola.utils.imports import check_xarray, is_xarray_available

if is_xarray_available():
    from xarray import DataArray, Dataset, Variable
else:
    DataArray, Dataset, Variable = None, None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class DataArrayAllCloseOperator(BaseAllCloseOperator[DataArray]):
    r"""Implements an allclose operator for ``xarray.DataArray``."""

    def __init__(self) -> None:
        check_xarray()

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: DataArray,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, DataArray):
            if show_difference:
                logger.info(f"object2 is not a xarray.DataArray: {type(object2)}")
            return False
        object_equal = (
            tester.allclose(
                object1.variable,
                object2.variable,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                show_difference=show_difference,
            )
            and objects_are_equal(
                object1.name,
                object2.name,
                show_difference=show_difference,
            )
            and objects_are_equal(
                object1.coords,
                object2.coords,
                show_difference=show_difference,
            )
        )
        if show_difference and not object_equal:
            logger.info(
                f"xarray.DataArrays are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class DataArrayEqualityOperator(BaseEqualityOperator[DataArray]):
    r"""Implements an equality operator for ``xarray.DataArray``."""

    def __init__(self) -> None:
        check_xarray()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: DataArray,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, DataArray):
            if show_difference:
                logger.info(f"object2 is not a xarray.DataArray: {type(object2)}")
            return False
        object_equal = (
            tester.equal(object1.variable, object2.variable, show_difference)
            and tester.equal(object1.name, object2.name, show_difference)
            and tester.equal(object1._coords, object2._coords, show_difference)
        )
        if show_difference and not object_equal:
            logger.info(
                f"xarray.DataArrays are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class DatasetAllCloseOperator(BaseAllCloseOperator[Dataset]):
    r"""Implements an allclose operator for ``xarray.Dataset``."""

    def __init__(self) -> None:
        check_xarray()

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Dataset,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, Dataset):
            if show_difference:
                logger.info(f"object2 is not a xarray.Dataset: {type(object2)}")
            return False
        object_equal = (
            tester.allclose(
                object1.data_vars,
                object2.data_vars,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                show_difference=show_difference,
            )
            and objects_are_equal(
                object1.coords,
                object2.coords,
                show_difference=show_difference,
            )
            and objects_are_equal(
                object1.attrs,
                object2.attrs,
                show_difference=show_difference,
            )
        )
        if show_difference and not object_equal:
            logger.info(f"xarray.Datasets are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


class DatasetEqualityOperator(BaseEqualityOperator[Dataset]):
    r"""Implements an equality operator for ``xarray.Dataset``."""

    def __init__(self) -> None:
        check_xarray()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Dataset,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, Dataset):
            if show_difference:
                logger.info(f"object2 is not a xarray.Dataset: {type(object2)}")
            return False
        object_equal = (
            tester.equal(object1.data_vars, object2.data_vars, show_difference)
            and tester.equal(object1.coords, object2.coords, show_difference)
            and tester.equal(object1.attrs, object2.attrs, show_difference)
        )
        if show_difference and not object_equal:
            logger.info(f"xarray.Datasets are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


class VariableAllCloseOperator(BaseAllCloseOperator[Variable]):
    r"""Implements an allclose operator for ``xarray.Variable``."""

    def __init__(self) -> None:
        check_xarray()

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Variable,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, Variable):
            if show_difference:
                logger.info(f"object2 is not a xarray.Variable: {type(object2)}")
            return False
        object_equal = (
            tester.allclose(
                object1.data,
                object2.data,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                show_difference=show_difference,
            )
            and objects_are_equal(
                object1.dims,
                object2.dims,
                show_difference=show_difference,
            )
            and objects_are_equal(
                object1.attrs,
                object2.attrs,
                show_difference=show_difference,
            )
        )
        if show_difference and not object_equal:
            logger.info(f"xarray.Variables are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


class VariableEqualityOperator(BaseEqualityOperator[Variable]):
    r"""Implements an equality operator for ``xarray.Variable``."""

    def __init__(self) -> None:
        check_xarray()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Variable,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, Variable):
            if show_difference:
                logger.info(f"object2 is not a xarray.Variable: {type(object2)}")
            return False
        object_equal = (
            tester.equal(object1.data, object2.data, show_difference)
            and tester.equal(object1.dims, object2.dims, show_difference)
            and tester.equal(object1.attrs, object2.attrs, show_difference)
        )
        if show_difference and not object_equal:
            logger.info(f"xarray.Variables are different\nobject1=\n{object1}\nobject2=\n{object2}")
        return object_equal


if is_xarray_available():  # pragma: no cover
    if not AllCloseTester.has_allclose_operator(DataArray):
        AllCloseTester.add_allclose_operator(DataArray, DataArrayAllCloseOperator())
    if not AllCloseTester.has_allclose_operator(Dataset):
        AllCloseTester.add_allclose_operator(Dataset, DatasetAllCloseOperator())
    if not AllCloseTester.has_allclose_operator(Variable):
        AllCloseTester.add_allclose_operator(Variable, VariableAllCloseOperator())

    if not EqualityTester.has_equality_operator(DataArray):
        EqualityTester.add_equality_operator(DataArray, DataArrayEqualityOperator())
    if not EqualityTester.has_equality_operator(Dataset):
        EqualityTester.add_equality_operator(Dataset, DatasetEqualityOperator())
    if not EqualityTester.has_equality_operator(Variable):
        EqualityTester.add_equality_operator(Variable, VariableEqualityOperator())
