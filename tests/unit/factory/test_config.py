from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Union

import pytest

from coola.factory import OBJECT_TARGET, is_object_config

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


def create_list() -> list:
    """Helper function that returns a list."""
    return [1, 2, 3, 4]


def create_list_union() -> Union[list, tuple]:  # noqa: UP007
    """Helper function that returns a list with union type hint."""
    return [1, 2, 3, 4]


# The missing return annotation is the point of this helper, so ANN201 is ignored on purpose.
def create_list_without_type_hint():  # noqa: ANN201
    """Helper function without type hint."""
    return [1, 2, 3, 4]


def create_list_union2() -> list | tuple:
    """Helper function with modern union syntax."""
    return [1, 2, 3, 4]


def create_undefined_return_type() -> UndefinedType:  # noqa: F821  # ty: ignore[unresolved-reference]
    """Helper function whose return type hint cannot be resolved."""
    return []


class Interface(ABC):
    """An abstract class with a virtual subclass."""

    @abstractmethod
    def run(self) -> None:
        """Run the object."""


class VirtualSubclass:
    """A class registered as a virtual subclass of ``Interface``."""


Interface.register(VirtualSubclass)


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


def test_is_object_config_empty_config_str() -> None:
    """Test is_object_config with completely empty config."""
    assert not is_object_config({}, str)


def test_is_object_config_empty_config_dict() -> None:
    assert not is_object_config({}, dict)


def test_is_object_config_empty_config_object() -> None:
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


def test_is_object_config_builtin_str_str() -> None:
    """Test is_object_config with builtin string type."""
    assert is_object_config({OBJECT_TARGET: "builtins.str"}, str)


def test_is_object_config_builtin_str_object() -> None:
    assert is_object_config({OBJECT_TARGET: "builtins.str"}, object)


def test_is_object_config_builtin_list_list() -> None:
    """Test is_object_config with builtin list type."""
    assert is_object_config({OBJECT_TARGET: "builtins.list"}, list)


def test_is_object_config_builtin_list_object() -> None:
    assert is_object_config({OBJECT_TARGET: "builtins.list"}, object)


def test_is_object_config_builtin_dict_dict() -> None:
    """Test is_object_config with builtin dict type."""
    assert is_object_config({OBJECT_TARGET: "builtins.dict"}, dict)


def test_is_object_config_builtin_dict_object() -> None:
    assert is_object_config({OBJECT_TARGET: "builtins.dict"}, object)


def test_is_object_config_with_object_class_int() -> None:
    """Test is_object_config with object base class."""
    # Any class should match object since everything inherits from object
    assert is_object_config({OBJECT_TARGET: "builtins.int"}, object)


def test_is_object_config_with_object_class_str() -> None:
    assert is_object_config({OBJECT_TARGET: "builtins.str"}, object)


def test_is_object_config_with_object_class_counter() -> None:
    assert is_object_config({OBJECT_TARGET: "collections.Counter"}, object)


def test_is_object_config_counter_not_list() -> None:
    """Test is_object_config correctly identifies Counter is not a
    list."""
    assert not is_object_config({OBJECT_TARGET: "collections.Counter"}, list)


def test_is_object_config_int_not_str() -> None:
    """Test is_object_config correctly identifies int is not str."""
    assert not is_object_config({OBJECT_TARGET: "builtins.int"}, str)


def test_is_object_config_with_invalid_target() -> None:
    """Test is_object_config with non-existent target raises ImportError
    if the module is imported."""
    with pytest.raises(ImportError, match="The target object does not exist"):
        is_object_config(
            {OBJECT_TARGET: "non_existent_module.NonExistentClass"}, object, import_target=True
        )


def test_is_object_config_with_target_in_a_module_that_is_not_imported() -> None:
    """Test is_object_config returns False without importing the
    module."""
    assert not is_object_config({OBJECT_TARGET: "non_existent_module.NonExistentClass"}, object)


def test_is_object_config_with_missing_attribute_in_an_imported_module() -> None:
    with pytest.raises(ImportError, match="The target object does not exist"):
        is_object_config({OBJECT_TARGET: "collections.NonExistentClass"}, object)


def test_is_object_config_with_malformed_target() -> None:
    """Test is_object_config with malformed target string."""
    with pytest.raises(ImportError, match="The target object does not exist"):
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


def test_is_object_config_function_with_unresolvable_return_type_hint() -> None:
    # ``get_type_hints`` raises a ``NameError``: the config is not for ``cls``
    assert not is_object_config(
        {OBJECT_TARGET: "tests.unit.factory.test_config.create_undefined_return_type"}, list
    )


def test_is_object_config_virtual_subclass() -> None:
    assert is_object_config(
        {OBJECT_TARGET: "tests.unit.factory.test_config.VirtualSubclass"}, Interface
    )


def test_is_object_config_not_a_virtual_subclass() -> None:
    assert not is_object_config({OBJECT_TARGET: "builtins.int"}, Interface)


def test_is_object_config_allowed_prefixes_match_module() -> None:
    config = {OBJECT_TARGET: "collections.Counter"}
    assert is_object_config(config, Counter, allowed_prefixes=["collections"])


def test_is_object_config_allowed_prefixes_match_class() -> None:
    config = {OBJECT_TARGET: "collections.Counter"}
    assert is_object_config(config, Counter, allowed_prefixes=["os", "collections.Counter"])


def test_is_object_config_allowed_prefixes_no_match() -> None:
    with pytest.raises(ImportError, match="not allowed"):
        is_object_config({OBJECT_TARGET: "collections.Counter"}, Counter, allowed_prefixes=["os"])


def test_is_object_config_allowed_prefixes_must_match_a_whole_module_name() -> None:
    with pytest.raises(ImportError, match="not allowed"):
        is_object_config(
            {OBJECT_TARGET: "collections.Counter"}, Counter, allowed_prefixes=["collection"]
        )


def test_is_object_config_allowed_prefixes_checked_before_the_import() -> None:
    # The module is not imported if it is not allowed: the error is not a "module not found".
    with pytest.raises(ImportError, match="not allowed"):
        is_object_config(
            {OBJECT_TARGET: "module_that_does_not_exist.Class"},
            object,
            allowed_prefixes=["collections"],
        )


@pytest.fixture
def side_effect_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    r"""Create a module that writes a file when it is imported."""
    marker = tmp_path.joinpath("imported.txt")
    tmp_path.joinpath("coola_side_effect_module.py").write_text(
        f"from pathlib import Path\n\nPath({str(marker)!r}).write_text('imported')\n\n\n"
        "class Thing:\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("coola_side_effect_module", None)
    yield marker
    sys.modules.pop("coola_side_effect_module", None)


def test_is_object_config_does_not_import_the_module_of_the_target(
    side_effect_module: Path,
) -> None:
    is_object_config({OBJECT_TARGET: "coola_side_effect_module.Thing"}, object)
    assert not side_effect_module.exists()


@pytest.mark.usefixtures("side_effect_module")
def test_is_object_config_target_in_a_module_that_is_not_imported_is_false() -> None:
    assert not is_object_config({OBJECT_TARGET: "coola_side_effect_module.Thing"}, object)


def test_is_object_config_import_target_imports_the_module(side_effect_module: Path) -> None:
    is_object_config({OBJECT_TARGET: "coola_side_effect_module.Thing"}, object, import_target=True)
    assert side_effect_module.exists()


@pytest.mark.usefixtures("side_effect_module")
def test_is_object_config_import_target_true() -> None:
    assert is_object_config(
        {OBJECT_TARGET: "coola_side_effect_module.Thing"}, object, import_target=True
    )


def test_is_object_config_allowed_prefixes_import_the_module(side_effect_module: Path) -> None:
    is_object_config(
        {OBJECT_TARGET: "coola_side_effect_module.Thing"},
        object,
        allowed_prefixes=["coola_side_effect_module"],
    )
    assert side_effect_module.exists()


@pytest.mark.usefixtures("side_effect_module")
def test_is_object_config_target_in_an_imported_module_is_true_without_import_target() -> None:
    importlib.import_module("coola_side_effect_module")
    assert is_object_config({OBJECT_TARGET: "coola_side_effect_module.Thing"}, object)


@pytest.mark.usefixtures("side_effect_module")
def test_is_object_config_module_name_target_is_not_imported() -> None:
    assert not is_object_config({OBJECT_TARGET: "coola_side_effect_module"}, object)


def test_is_object_config_target_that_is_not_a_string() -> None:
    assert not is_object_config({OBJECT_TARGET: 1}, object)


def test_is_object_config_target_that_is_not_a_string_with_import_target() -> None:
    assert not is_object_config({OBJECT_TARGET: 5}, int, import_target=True)


def test_is_object_config_allowed_prefixes_without_target() -> None:
    assert not is_object_config({}, object, allowed_prefixes=["collections"])


def create_list_invalid_annotation() -> list:
    """Helper function whose return annotation is not valid Python."""
    return [1, 2, 3, 4]


create_list_invalid_annotation.__annotations__["return"] = "list["


def test_is_object_config_invalid_string_annotation() -> None:
    config = {OBJECT_TARGET: f"{__name__}.create_list_invalid_annotation"}
    assert not is_object_config(config, list)


@pytest.mark.parametrize("config", ["x", None, 1, ["_target_"]])
def test_is_object_config_config_is_not_a_dict(config: object) -> None:
    assert not is_object_config(config, Counter)  # ty: ignore[invalid-argument-type]


def create_list_generic() -> list[int]:
    """Helper function that returns a list with a generic type hint."""
    return [1, 2, 3, 4]


def create_list_or_none_generic() -> list[int] | None:
    """Helper function with a union of a generic alias and None."""
    return [1, 2, 3, 4]


def test_is_object_config_function_generic_return_type() -> None:
    assert is_object_config({OBJECT_TARGET: f"{__name__}.create_list_generic"}, list)


def test_is_object_config_function_generic_return_type_wrong_class() -> None:
    assert not is_object_config({OBJECT_TARGET: f"{__name__}.create_list_generic"}, dict)


def test_is_object_config_function_generic_union_return_type() -> None:
    assert is_object_config({OBJECT_TARGET: f"{__name__}.create_list_or_none_generic"}, list)
