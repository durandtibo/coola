from __future__ import annotations

import warnings
from dataclasses import dataclass
from decimal import Decimal

import pytest

from coola.registry import TypeNotRegisteredError, TypeRegistry
from coola.rounding import (
    UnsupportedTypeError,
    get_default_registry,
    register_rounders,
    round_values,
)
from coola.rounding.common import Rounder, RoundMode
from coola.rounding.interface import _register_default_rounders, _round_leaf, _round_object
from coola.rounding.native import round_float
from coola.testing.marks import (
    numpy_available,
    pandas_available,
    polars_available,
    torch_available,
)
from coola.utils.imports import (
    MissingPackageError,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
)

if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_torch_available():
    import torch

#####################################
#     Tests for round_values        #
#####################################


def test_round_values_default() -> None:
    assert round_values(2.567) == 3.0


def test_round_values_float() -> None:
    assert round_values(2.567, mode="floor", decimals=2) == 2.56


def test_round_values_float_type() -> None:
    assert type(round_values(2.5)) is float


def test_round_values_int() -> None:
    assert round_values(3, mode="ceil", decimals=2) == 3


@pytest.mark.parametrize(("mode", "expected"), [("round", 1.2), ("floor", 1.2), ("ceil", 1.3)])
def test_round_values_modes(mode: RoundMode, expected: float) -> None:
    assert round_values(1.234, mode=mode, decimals=1) == expected


def test_round_values_incorrect_mode() -> None:
    with pytest.raises(ValueError, match=r"mode must be one of"):
        round_values(1.5, mode="trunc")  # ty: ignore[invalid-argument-type]


def test_round_values_negative_decimals() -> None:
    with pytest.raises(ValueError, match=r"decimals must be greater than or equal to 0"):
        round_values(1.5, decimals=-1)


def test_round_values_decimals_not_int() -> None:
    with pytest.raises(TypeError, match=r"decimals must be an int"):
        round_values(1.5, decimals=1.0)  # ty: ignore[invalid-argument-type]


def test_round_values_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match=r"positional"):
        round_values(1.5, "floor")  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize("data", ["abc", None, Decimal("1.5"), [1.5], {"a": 1.5}, object()])
def test_round_values_unsupported(data: object) -> None:
    with pytest.raises(UnsupportedTypeError, match=r"Cannot round an object of type"):
        round_values(data)


def test_round_values_recursive_nested() -> None:
    data = {"a": [1.234, "x", None], "b": (5.678, 3), "c": {2.5}}
    assert round_values(data, mode="floor", decimals=1, recursive=True) == {
        "a": [1.2, "x", None],
        "b": (5.6, 3),
        "c": {2.5},
    }


def test_round_values_recursive_does_not_modify_input() -> None:
    data = {"a": [1.234]}
    round_values(data, recursive=True)
    assert data == {"a": [1.234]}


def test_round_values_recursive_leaf() -> None:
    assert round_values(2.567, decimals=1, recursive=True) == 2.6


@dataclass
class Point:
    x: float


def test_round_values_recursive_enters_the_dataclass() -> None:
    assert round_values(Point(1.234), decimals=1, recursive=True) == Point(1.2)


@numpy_available
def test_round_values_numpy_array() -> None:
    assert round_values(np.array([1.234, 5.678]), mode="floor", decimals=1).tolist() == [1.2, 5.6]


@numpy_available
def test_round_values_numpy_float64_scalar() -> None:
    assert type(round_values(np.float64(1.234), decimals=1)) is np.float64


@numpy_available
def test_round_values_recursive_numpy() -> None:
    result = round_values([np.array([1.26])], mode="floor", decimals=1, recursive=True)
    assert result[0].tolist() == [1.2]


@torch_available
def test_round_values_torch_tensor() -> None:
    result = round_values(
        torch.tensor([1.234, 5.678], dtype=torch.float64), mode="ceil", decimals=1
    )
    assert result.tolist() == [1.3, 5.7]


@torch_available
def test_round_values_torch_tensor_subclass() -> None:
    result = round_values(torch.nn.Parameter(torch.tensor([1.5, 2.5])))
    assert result.tolist() == [2.0, 2.0]


@pandas_available
def test_round_values_pandas_dataframe() -> None:
    result = round_values(pd.DataFrame({"a": [1.234], "b": ["x"]}), decimals=1)
    assert result.to_dict("list") == {"a": [1.2], "b": ["x"]}


@polars_available
def test_round_values_polars_dataframe() -> None:
    result = round_values(pl.DataFrame({"a": [1.234], "b": ["x"]}), decimals=1)
    assert result.to_dict(as_series=False) == {"a": [1.2], "b": ["x"]}


@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
@torch_available
@numpy_available
def test_round_values_numpy_and_torch_agree(mode: RoundMode) -> None:
    values = [0.29, 0.07, -2.567, 2.5, 1.005, 1e-3]
    expected = round_values(np.array(values), mode=mode, decimals=2).tolist()
    assert (
        round_values(torch.tensor(values, dtype=torch.float64), mode=mode, decimals=2).tolist()
        == expected
    )


@numpy_available
def test_round_values_numpy_integer_decimals() -> None:
    assert round_values(1.26, mode="floor", decimals=np.int64(1)) == 1.2  # ty: ignore[invalid-argument-type]


###################################################
#     Tests for the registry and the error class  #
###################################################


class Meters:
    def __init__(self, value: float) -> None:
        self.value = value


def round_meters(data: Meters, mode: RoundMode, decimals: int) -> Meters:
    return Meters(round_values(data.value, mode=mode, decimals=decimals))


def raise_missing_package(data: object, mode: RoundMode, decimals: int) -> object:
    del data, mode, decimals
    msg = "No module named 'fake_package'"
    raise MissingPackageError(msg, name="fake_package")


def raise_value_error(data: object, mode: RoundMode, decimals: int) -> object:
    del data, mode, decimals
    msg = "bug in the rounder"
    raise ValueError(msg)


def make_registry(rounder: Rounder) -> TypeRegistry[Rounder]:
    registry = TypeRegistry[Rounder]()
    registry.register(Meters, rounder)
    return registry


def test_unsupported_type_error_is_a_type_error() -> None:
    assert issubclass(UnsupportedTypeError, TypeError)


def test_round_values_registry_is_used() -> None:
    result = round_values(
        Meters(1.26), mode="floor", decimals=1, registry=make_registry(round_meters)
    )
    assert result.value == 1.2


def test_round_values_registry_replaces_the_default_registry() -> None:
    with pytest.raises(UnsupportedTypeError, match=r"Cannot round an object of type float"):
        round_values(1.5, registry=make_registry(round_meters))


def test_round_values_registry_miss_message_names_the_remedy() -> None:
    with pytest.raises(UnsupportedTypeError, match=r"register a rounder with register_rounders"):
        round_values("abc")


def test_round_values_registry_miss_cause() -> None:
    with pytest.raises(
        UnsupportedTypeError, match="Cannot round an object of type str"
    ) as exc_info:
        round_values("abc")
    assert isinstance(exc_info.value.__cause__, TypeNotRegisteredError)


@pytest.mark.usefixtures("isolated_default_registries")
def test_register_rounders() -> None:
    register_rounders({Meters: round_meters})
    assert round_values(Meters(1.26), mode="ceil", decimals=1).value == 1.3


@pytest.mark.usefixtures("isolated_default_registries")
def test_register_rounders_twice_raises() -> None:
    register_rounders({Meters: round_meters})
    with pytest.raises(Exception, match=r"already registered"):
        register_rounders({Meters: round_meters})


@pytest.mark.usefixtures("isolated_default_registries")
def test_register_rounders_exist_ok() -> None:
    register_rounders({Meters: round_meters})
    register_rounders({Meters: raise_value_error}, exist_ok=True)
    with pytest.raises(ValueError, match=r"bug in the rounder"):
        round_values(Meters(1.0))


@pytest.mark.usefixtures("isolated_default_registries")
def test_register_rounders_subclass_uses_the_rounder_of_the_parent() -> None:
    class Kilometers(Meters):
        pass

    register_rounders({Meters: round_meters})
    assert round_values(Kilometers(1.26), mode="floor", decimals=1).value == 1.2


def test_get_default_registry_is_a_singleton() -> None:
    assert get_default_registry() is get_default_registry()


def test_get_default_registry_has_the_native_rounders() -> None:
    assert get_default_registry().has(float)


def test_get_default_registry_bool_uses_the_int_rounder() -> None:
    assert get_default_registry().resolve(bool) is get_default_registry().resolve(int)


@numpy_available
def test_get_default_registry_loads_numpy_rounders_lazily() -> None:
    assert get_default_registry().has(np.ndarray)


###############################################
#     Tests for the validation of arguments   #
###############################################


def test_round_values_validates_mode_before_decimals() -> None:
    with pytest.raises(ValueError, match=r"mode must be one of"):
        round_values(1.5, mode="trunc", decimals=-1)  # ty: ignore[invalid-argument-type]


def test_round_values_validates_before_any_rounder_runs() -> None:
    with pytest.raises(ValueError, match=r"decimals must be greater than or equal to 0"):
        round_values(Meters(1.5), decimals=-1, registry=make_registry(raise_value_error))


@pytest.mark.parametrize("decimals", [True, 1.0])
def test_round_values_decimals_bool_or_float_rejected(decimals: object) -> None:
    with pytest.raises(TypeError, match=r"decimals must be an int"):
        round_values(1.5, decimals=decimals)  # ty: ignore[invalid-argument-type]


def test_round_values_invalid_on_missing_package() -> None:
    with pytest.raises(ValueError, match=r"on_missing_package must be one of"):
        round_values(1.5, on_missing_package="skip")  # ty: ignore[invalid-argument-type]


def test_round_values_invalid_on_missing_package_raises_before_any_rounder_runs() -> None:
    with pytest.raises(ValueError, match=r"on_missing_package must be one of"):
        round_values(
            Meters(1.5),
            on_missing_package="skip",  # ty: ignore[invalid-argument-type]
            registry=make_registry(raise_value_error),
        )


#########################################
#     Tests for on_missing_package      #
#########################################


def test_round_values_missing_package_raise() -> None:
    with pytest.raises(UnsupportedTypeError, match=r"the package 'fake_package' is not installed"):
        round_values(Meters(1.5), registry=make_registry(raise_missing_package))


def test_round_values_missing_package_raise_is_the_default_and_names_the_type() -> None:
    with pytest.raises(UnsupportedTypeError, match=r"type Meters"):
        round_values(Meters(1.5), registry=make_registry(raise_missing_package))


def test_round_values_missing_package_raise_cause() -> None:
    with pytest.raises(UnsupportedTypeError, match="fake_package") as exc_info:
        round_values(Meters(1.5), registry=make_registry(raise_missing_package))
    assert isinstance(exc_info.value.__cause__, MissingPackageError)


def test_round_values_missing_package_warn_warns() -> None:
    with pytest.warns(UserWarning, match=r"type Meters is not rounded.*'fake_package'"):
        round_values(
            Meters(1.5), on_missing_package="warn", registry=make_registry(raise_missing_package)
        )


def test_round_values_missing_package_warn_returns_the_object() -> None:
    data = Meters(1.5)
    with pytest.warns(UserWarning, match=r"is not rounded"):
        result = round_values(
            data, on_missing_package="warn", registry=make_registry(raise_missing_package)
        )
    assert result is data


def test_round_values_missing_package_ignore_returns_the_object_silently() -> None:
    data = Meters(1.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = round_values(
            data, on_missing_package="ignore", registry=make_registry(raise_missing_package)
        )
    assert result is data


@pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
def test_round_values_no_rounder_raises_whatever_the_policy(policy: str) -> None:
    with pytest.raises(UnsupportedTypeError, match=r"Cannot round an object of type str"):
        round_values("abc", on_missing_package=policy)  # ty: ignore[invalid-argument-type]


def test_round_values_missing_package_recursive_raise_stops_the_walk() -> None:
    with pytest.raises(UnsupportedTypeError, match=r"fake_package"):
        round_values(
            [1.26, Meters(1.5)],
            recursive=True,
            registry=_registry_with_float_and(raise_missing_package),
        )


def test_round_values_missing_package_recursive_ignore_keeps_the_leaf() -> None:
    data = Meters(1.5)
    result = round_values(
        [1.26, data],
        mode="floor",
        decimals=1,
        recursive=True,
        on_missing_package="ignore",
        registry=_registry_with_float_and(raise_missing_package),
    )
    assert result == [1.2, data]


def test_round_values_missing_package_recursive_warn_warns_once_per_leaf() -> None:
    with pytest.warns(UserWarning, match=r"is not rounded") as record:
        round_values(
            [Meters(1.5), Meters(2.5)],
            recursive=True,
            on_missing_package="warn",
            registry=_registry_with_float_and(raise_missing_package),
        )
    assert len(record) == 2


def _registry_with_float_and(rounder: Rounder) -> TypeRegistry[Rounder]:
    registry = make_registry(rounder)
    registry.register(float, round_float)
    return registry


######################################
#     Tests for recursive=True       #
######################################


def test_round_values_recursive_keeps_an_unsupported_leaf() -> None:
    assert round_values(["abc", 1.26], mode="floor", decimals=1, recursive=True) == ["abc", 1.2]


def test_round_values_recursive_propagates_the_error_of_a_rounder() -> None:
    """Regression test: a ``ValueError`` of a rounder was hidden."""
    with pytest.raises(ValueError, match=r"bug in the rounder"):
        round_values([Meters(1.5)], recursive=True, registry=make_registry(raise_value_error))


def test_round_values_recursive_propagates_a_type_error_of_a_rounder() -> None:
    def raise_type_error(data: object, mode: RoundMode, decimals: int) -> object:
        del data, mode, decimals
        msg = "bug in the rounder"
        raise TypeError(msg)

    with pytest.raises(TypeError, match=r"bug in the rounder"):
        round_values([Meters(1.5)], recursive=True, registry=make_registry(raise_type_error))


def test_round_values_recursive_nested_output() -> None:
    data = {"a": [1.234, "x", None, 3], "b": (5.678, True), "c": {"d": [Meters]}}
    assert round_values(data, mode="ceil", decimals=1, recursive=True) == {
        "a": [1.3, "x", None, 3],
        "b": (5.7, True),
        "c": {"d": [Meters]},
    }


def test_round_leaf_keeps_an_unsupported_leaf() -> None:
    assert _round_leaf("abc", get_default_registry(), "round", 0, "raise") == "abc"


def test_round_leaf_rounds_a_supported_leaf() -> None:
    assert _round_leaf(1.26, get_default_registry(), "floor", 1, "raise") == 1.2


def test_round_object_unsupported_type() -> None:
    with pytest.raises(UnsupportedTypeError, match=r"Cannot round an object of type str"):
        _round_object("abc", get_default_registry(), "round", 0, "raise")


def test_round_object_rounds() -> None:
    assert _round_object(1.26, get_default_registry(), "ceil", 1, "raise") == 1.3


#################################################
#     Tests for _register_default_rounders     #
#################################################


@numpy_available
def test_register_default_rounders_skips_missing_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("coola.rounding.interface.package_available", lambda _: False)
    registry = TypeRegistry()
    _register_default_rounders(registry)
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        registry.resolve(np.ndarray)


@numpy_available
def test_register_default_rounders_registers_numpy_rounder() -> None:
    registry = TypeRegistry()
    _register_default_rounders(registry)
    assert registry.resolve(np.ndarray) is not None


def test_register_default_rounders_registers_native_rounders() -> None:
    registry = TypeRegistry()
    _register_default_rounders(registry)
    assert registry.resolve(float) is round_float
