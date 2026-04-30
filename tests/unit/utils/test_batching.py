from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.utils.batching import batchify

if TYPE_CHECKING:
    from collections.abc import Iterator


def infinite() -> Iterator[int]:
    n = 0
    while True:
        yield n
        n += 1


##############################
#     Tests for batchify     #
##############################


def test_batchify_even_split() -> None:
    assert list(batchify([1, 2, 3, 4], size=2)) == [(1, 2), (3, 4)]


def test_batchify_uneven_split() -> None:
    assert list(batchify([1, 2, 3, 4, 5], size=2)) == [(1, 2), (3, 4), (5,)]


def test_batchify_size_larger_than_items() -> None:
    assert list(batchify([1, 2, 3], size=10)) == [(1, 2, 3)]


def test_batchify_size_equals_length() -> None:
    assert list(batchify([1, 2, 3], size=3)) == [(1, 2, 3)]


def test_batchify_size_one() -> None:
    assert list(batchify([1, 2, 3], size=1)) == [(1,), (2,), (3,)]


def test_batchify_empty_sequence() -> None:
    assert list(batchify([], size=3)) == []


def test_batchify_single_item() -> None:
    assert list(batchify([42], size=3)) == [(42,)]


def test_batchify_with_generator() -> None:
    assert list(batchify(range(6), size=2)) == [(0, 1), (2, 3), (4, 5)]


def test_batchify_with_iterator() -> None:
    assert list(batchify(iter(range(6)), size=2)) == [(0, 1), (2, 3), (4, 5)]


def test_batchify_with_strings() -> None:
    assert list(batchify("abcde", size=2)) == [("a", "b"), ("c", "d"), ("e",)]


def test_batchify_with_mixed_types() -> None:
    assert list(batchify([1, "a", None, 3.14], size=2)) == [(1, "a"), (None, 3.14)]


def test_batchify_returns_iterator() -> None:
    result = batchify([1, 2, 3], size=2)
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")


def test_batchify_is_lazy() -> None:

    it = batchify(infinite(), size=3)
    assert next(it) == (0, 1, 2)
    assert next(it) == (3, 4, 5)


def test_batchify_invalid_size_zero() -> None:
    with pytest.raises(ValueError, match="size must be >= 1"):
        list(batchify([1, 2, 3], size=0))


def test_batchify_invalid_size_negative() -> None:
    with pytest.raises(ValueError, match="size must be >= 1"):
        list(batchify([1, 2, 3], size=-5))


def test_batchify_invalid_size_raises_before_iteration() -> None:
    """Size validation fires eagerly, before any item is consumed."""
    with pytest.raises(ValueError, match="size must be >= 1"):
        list(batchify(infinite(), size=0))
