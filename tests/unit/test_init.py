from __future__ import annotations

import importlib

import pytest

import coola


def test_version_exists() -> None:
    assert hasattr(coola, "__version__")


def test_version_is_a_str() -> None:
    assert isinstance(coola.__version__, str)


def test_version_is_not_empty() -> None:
    assert coola.__version__


def test_version_is_cached_in_the_module() -> None:
    coola.__version__  # noqa: B018
    assert "__version__" in vars(coola)


def test_dir_lists_the_version() -> None:
    assert "__version__" in dir(coola)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        coola.missing  # noqa: B018


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("assert_objects_allclose", "coola.equality"),
        ("assert_objects_equal", "coola.equality"),
        ("compare", "coola.equality"),
        ("hash_object", "coola.hashing"),
        ("objects_are_allclose", "coola.equality"),
        ("objects_are_equal", "coola.equality"),
        ("summarize", "coola.summary"),
    ],
)
def test_entry_point_is_the_object_of_the_subpackage(name: str, module: str) -> None:
    assert getattr(coola, name) is getattr(importlib.import_module(module), name)


def test_entry_point_is_cached_in_the_module() -> None:
    coola.objects_are_equal  # noqa: B018
    assert "objects_are_equal" in vars(coola)


def test_dir_lists_the_entry_points_that_are_not_imported_yet() -> None:
    assert "summarize" in dir(coola)


def test_objects_are_equal() -> None:
    assert coola.objects_are_equal([1, 2], [1, 2])
