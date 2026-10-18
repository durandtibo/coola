from __future__ import annotations

import warnings

import pytest

from coola.utils.deprecation import deprecated_alias


def add(a: int, b: int = 1) -> int:
    """Add two numbers."""
    return a + b


###################################
#     Tests for deprecated_alias  #
###################################


def test_deprecated_alias_returns_the_result_of_the_function() -> None:
    alias = deprecated_alias(add, "old_add")
    with pytest.warns(DeprecationWarning, match="old_add"):
        result = alias(1, b=2)
    assert result == 3


def test_deprecated_alias_passes_positional_and_keyword_arguments() -> None:
    alias = deprecated_alias(add, "old_add")
    with pytest.warns(DeprecationWarning, match="old_add"):
        default = alias(2)
    with pytest.warns(DeprecationWarning, match="old_add"):
        positional = alias(2, 5)
    assert (default, positional) == (3, 7)


def test_deprecated_alias_warning_message() -> None:
    alias = deprecated_alias(add, "old_add")
    with pytest.warns(
        DeprecationWarning, match="`old_add` is deprecated.*`tests.unit.utils.test_deprecation.add`"
    ):
        alias(1)


def test_deprecated_alias_warning_points_to_the_caller() -> None:
    alias = deprecated_alias(add, "old_add")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias(1)
    assert caught[0].filename == __file__


def test_deprecated_alias_name_and_doc() -> None:
    alias = deprecated_alias(add, "old_add")
    assert alias.__name__ == "old_add"
    assert alias.__qualname__ == "old_add"
    assert alias.__doc__ is not None
    assert "Deprecated alias of ``add``" in alias.__doc__


def test_deprecated_alias_does_not_modify_the_function() -> None:
    deprecated_alias(add, "old_add")
    assert add.__name__ == "add"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert add(1) == 2


def test_deprecated_alias_removes_interface_suffix_from_module_in_message() -> None:
    def func() -> None:
        pass

    func.__module__ = "coola.pkg.interface"
    alias = deprecated_alias(func, "old_func")
    with pytest.warns(DeprecationWarning, match=r"`coola\.pkg\.func`"):
        alias()


def test_deprecated_alias_callable_without_name() -> None:
    class Callee:
        def __call__(self) -> int:
            return 1

    alias = deprecated_alias(Callee(), "old_callee")
    with pytest.warns(DeprecationWarning, match="old_callee"):
        assert alias() == 1
