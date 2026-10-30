from __future__ import annotations

import pytest

import coola


def test_version_exists() -> None:
    assert hasattr(coola, "__version__")


def test_version_is_a_str() -> None:
    assert isinstance(coola.__version__, str)


def test_version_is_not_empty() -> None:
    assert coola.__version__


def test_version_is_cached_in_the_module() -> None:
    _ = coola.__version__
    assert "__version__" in vars(coola)


def test_dir_lists_the_version() -> None:
    assert "__version__" in dir(coola)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        _ = coola.missing
