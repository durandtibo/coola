from __future__ import annotations

import coola


def test_version_exists() -> None:
    assert hasattr(coola, "__version__")
    assert isinstance(coola.__version__, str)


def test_objects_are_equal_exported() -> None:
    assert hasattr(coola, "objects_are_equal")
    assert callable(coola.objects_are_equal)


def test_objects_are_allclose_exported() -> None:
    assert hasattr(coola, "objects_are_allclose")
    assert callable(coola.objects_are_allclose)


def test_summarize_exported() -> None:
    assert hasattr(coola, "summarize")
    assert callable(coola.summarize)


def test_objects_are_equal_works() -> None:
    assert coola.objects_are_equal([1, 2, 3], [1, 2, 3])
    assert not coola.objects_are_equal([1, 2, 3], [1, 2, 4])


def test_objects_are_allclose_works() -> None:
    assert coola.objects_are_allclose([1.0, 2.0], [1.0 + 1e-9, 2.0])
    assert not coola.objects_are_allclose([1.0, 2.0], [1.0, 3.0])


def test_summarize_works() -> None:
    summary = coola.summarize({"a": [1, 2, 3]})
    assert isinstance(summary, str)
    assert "dict" in summary
