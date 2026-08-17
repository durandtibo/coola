r"""Benchmarks for ``objects_are_equal`` and ``objects_are_allclose``.

These benchmarks are not part of the regular unit/integration test suite
(``inv unit-test`` and ``inv integration-test`` target ``tests/unit`` and
``tests/integration`` explicitly, so this directory is skipped). Run them
explicitly with:

    inv benchmark

They exist to catch performance regressions in the equality comparison
machinery (tester/handler dispatch) as the codebase evolves, especially for
large or deeply nested objects.
"""

from __future__ import annotations

from coola.equality import objects_are_allclose, objects_are_equal


def make_wide_nested_object(width: int, depth: int) -> dict:
    r"""Create a dict with ``width`` keys at each of ``depth`` nested
    levels."""
    obj = {f"key{i}": float(i) for i in range(width)}
    for _ in range(depth):
        obj = {f"level{i}": dict(obj) for i in range(width)}
    return obj


def make_deep_nested_object(depth: int) -> dict | int:
    r"""Create a dict nested ``depth`` levels deep."""
    obj: dict | int = 0
    for _ in range(depth):
        obj = {"child": obj}
    return obj


def test_objects_are_equal_flat_list_benchmark(benchmark) -> None:  # noqa: ANN001
    data = list(range(10_000))
    other = list(range(10_000))
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_equal_wide_nested_benchmark(benchmark) -> None:  # noqa: ANN001
    data = make_wide_nested_object(width=10, depth=3)
    other = make_wide_nested_object(width=10, depth=3)
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_equal_deep_nested_benchmark(benchmark) -> None:  # noqa: ANN001
    data = make_deep_nested_object(depth=50)
    other = make_deep_nested_object(depth=50)
    assert benchmark(objects_are_equal, data, other)


def test_objects_are_allclose_flat_list_benchmark(benchmark) -> None:  # noqa: ANN001
    data = [float(i) for i in range(10_000)]
    other = [float(i) for i in range(10_000)]
    assert benchmark(objects_are_allclose, data, other)
