from __future__ import annotations

from collections import Counter
from typing import Union

import pytest

from coola.factory import OBJECT_TARGET, is_object_config


def create_list() -> list:
    """Helper function that returns a list."""
    return [1, 2, 3, 4]


def create_list_union() -> Union[list, tuple]:  # noqa: UP007
    """Helper function that returns a list with union type hint."""
    return [1, 2, 3, 4]


def create_list_without_type_hint():  # noqa: ANN201
    """Helper function without type hint."""
    return [1, 2, 3, 4]


def create_list_union2() -> list | tuple:
    """Helper function with modern union syntax."""
    return [1, 2, 3, 4]


######################################
#     Tests for is_object_config     #
######################################


@pytest.mark.parametrize(
    ("config", "cls"),
    [
        ({OBJECT_TARGET: "builtins.int"}, int),
        ({OBJECT_TARGET: "builtins.int"}, object),
        ({OBJECT_TARGET: "collections.Counter", "iterable": [1, 2, 1, 3]}, Counter),
        ({OBJECT_TARGET: "collections.Counter", "iterable": [1, 2, 1, 3]}, dict),
        ({OBJECT_TARGET: "collections.Counter", "iterable": [1, 2, 1, 3]}, object),
        ({OBJECT_TARGET: "tests.unit.factory.test_config.create_list"}, list),
        ({OBJECT_TARGET: "tests.unit.factory.test_config.create_list"}, object),
        ({OBJECT_TARGET: "tests.unit.factory.test_config.create_list_union"}, tuple),
        ({OBJECT_TARGET: "tests.unit.factory.test_config.create_list_union"}, list),
        ({OBJECT_TARGET: "tests.unit.factory.test_config.create_list_union"}, object),
    ],
)
def test_is_object_config_true(config: dict, cls: type[object]) -> None:
    """Test is_object_config returns True for valid configurations."""
    assert is_object_config(config, cls)


def test_is_object_config_true_union_type() -> None:
    """Test is_object_config with modern union type syntax."""
    assert is_object_config(
        {OBJECT_TARGET: "tests.unit.factory.test_config.create_list_union2"}, tuple
    )


def test_is_object_config_true_child_class() -> None:
    """Test is_object_config with a child class."""
    assert is_object_config({OBJECT_TARGET: "collections.Counter", "iterable": [1, 2, 1, 3]}, dict)


def test_is_object_config_false_missing_target() -> None:
    """Test is_object_config returns False when _target_ key is
    missing."""
    assert not is_object_config({}, int)


def test_is_object_config_false_incorrect_class() -> None:
    """Test is_object_config returns False for incorrect class."""
    assert not is_object_config({OBJECT_TARGET: "builtins.int"}, float)


def test_is_object_config_false_function_without_type_hint() -> None:
    """Test is_object_config returns False for functions without type
    hints."""
    assert not is_object_config(
        {OBJECT_TARGET: "tests.unit.factory.test_config.create_list_without_type_hint"}, float
    )


def test_is_object_config_empty_config() -> None:
    """Test is_object_config with completely empty config."""
    assert not is_object_config({}, str)
    assert not is_object_config({}, dict)
    assert not is_object_config({}, object)


def test_is_object_config_none_target() -> None:
    """Test is_object_config when _target_ is None."""
    assert not is_object_config({OBJECT_TARGET: None}, int)


def test_is_object_config_with_extra_keys() -> None:
    """Test is_object_config with extra configuration keys."""
    config = {
        OBJECT_TARGET: "collections.Counter",
        "iterable": [1, 2, 3],
        "extra_key": "extra_value",
    }
    assert is_object_config(config, Counter)


def test_is_object_config_builtin_str() -> None:
    """Test is_object_config with builtin string type."""
    assert is_object_config({OBJECT_TARGET: "builtins.str"}, str)
    assert is_object_config({OBJECT_TARGET: "builtins.str"}, object)


def test_is_object_config_builtin_list() -> None:
    """Test is_object_config with builtin list type."""
    assert is_object_config({OBJECT_TARGET: "builtins.list"}, list)
    assert is_object_config({OBJECT_TARGET: "builtins.list"}, object)


def test_is_object_config_builtin_dict() -> None:
    """Test is_object_config with builtin dict type."""
    assert is_object_config({OBJECT_TARGET: "builtins.dict"}, dict)
    assert is_object_config({OBJECT_TARGET: "builtins.dict"}, object)


def test_is_object_config_with_object_class() -> None:
    """Test is_object_config with object base class."""
    # Any class should match object since everything inherits from object
    assert is_object_config({OBJECT_TARGET: "builtins.int"}, object)
    assert is_object_config({OBJECT_TARGET: "builtins.str"}, object)
    assert is_object_config({OBJECT_TARGET: "collections.Counter"}, object)


def test_is_object_config_counter_not_list() -> None:
    """Test is_object_config correctly identifies Counter is not a
    list."""
    assert not is_object_config({OBJECT_TARGET: "collections.Counter"}, list)


def test_is_object_config_int_not_str() -> None:
    """Test is_object_config correctly identifies int is not str."""
    assert not is_object_config({OBJECT_TARGET: "builtins.int"}, str)


def test_is_object_config_with_invalid_target() -> None:
    """Test is_object_config with non-existent target raises
    ImportError."""
    with pytest.raises(ImportError):
        is_object_config({OBJECT_TARGET: "non_existent_module.NonExistentClass"}, object)


def test_is_object_config_with_malformed_target() -> None:
    """Test is_object_config with malformed target string."""
    with pytest.raises(ImportError):
        is_object_config({OBJECT_TARGET: "collections."}, object)


def test_is_object_config_function_return_list() -> None:
    """Test is_object_config with function returning list."""
    assert is_object_config({OBJECT_TARGET: "tests.unit.factory.test_config.create_list"}, list)


def test_is_object_config_function_not_matching_return_type() -> None:
    """Test is_object_config with function where return type doesn't
    match."""
    assert not is_object_config({OBJECT_TARGET: "tests.unit.factory.test_config.create_list"}, dict)


def test_is_object_config_target_is_not_a_class() -> None:
    """Test is_object_config returns False when the target resolves to a
    non-class object, e.g. a module-level constant."""
    assert not is_object_config({OBJECT_TARGET: "math.pi"}, object)
