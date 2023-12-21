from __future__ import annotations

__all__ = [
    "DataArrayAllCloseOperator",
    "DataArrayEqualityOperator",
    "DatasetAllCloseOperator",
    "DatasetEqualityOperator",
    "VariableAllCloseOperator",
    "VariableEqualityOperator",
    "get_mapping_allclose",
    "get_mapping_equality",
]

import logging
from typing import TYPE_CHECKING, Any

from coola.comparators.base import BaseAllCloseOperator, BaseEqualityOperator
from coola.comparison import objects_are_equal
from coola.utils.imports import check_xarray, is_xarray_available

if TYPE_CHECKING:
    from coola.testers import BaseAllCloseTester, BaseEqualityTester

if is_xarray_available():
    from xarray import DataArray, Dataset, Variable
else:
    DataArray, Dataset, Variable = None, None, None  # pragma: no cover

logger = logging.getLogger(__name__)


class DataArrayAllCloseOperator(BaseAllCloseOperator[DataArray]):
    r"""Implements an allclose operator for ``xarray.DataArray``."""

    def __init__(self) -> None:
        check_xarray()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

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

    def clone(self) -> DataArrayAllCloseOperator:
        return self.__class__()


class DataArrayEqualityOperator(BaseEqualityOperator[DataArray]):
    r"""Implements an equality operator for ``xarray.DataArray``."""

    def __init__(self) -> None:
        check_xarray()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> DataArrayEqualityOperator:
        return self.__class__()

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

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

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

    def clone(self) -> DatasetAllCloseOperator:
        return self.__class__()


class DatasetEqualityOperator(BaseEqualityOperator[Dataset]):
    r"""Implements an equality operator for ``xarray.Dataset``."""

    def __init__(self) -> None:
        check_xarray()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> DatasetEqualityOperator:
        return self.__class__()

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

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

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

    def clone(self) -> VariableAllCloseOperator:
        return self.__class__()


class VariableEqualityOperator(BaseEqualityOperator[Variable]):
    r"""Implements an equality operator for ``xarray.Variable``."""

    def __init__(self) -> None:
        check_xarray()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> VariableEqualityOperator:
        return self.__class__()

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


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Gets a default mapping between the types and the allclose
    operators.

    This function returns an empty dictionary if xarray is not
    installed.

    Returns:
        dict: The mapping between the types and the allclose
            operators.
    """
    if not is_xarray_available():
        return {}
    return {
        Dataset: DatasetAllCloseOperator(),
        DataArray: DataArrayAllCloseOperator(),
        Variable: VariableAllCloseOperator(),
    }


def get_mapping_equality() -> dict[type[object], BaseEqualityOperator]:
    r"""Gets a default mapping between the types and the equality
    operators.

    This function returns an empty dictionary if xarray is not
    installed.

    Returns:
        dict: The mapping between the types and the equality
            operators.
    """
    if not is_xarray_available():
        return {}
    return {
        Dataset: DatasetEqualityOperator(),
        DataArray: DataArrayEqualityOperator(),
        Variable: VariableEqualityOperator(),
    }
