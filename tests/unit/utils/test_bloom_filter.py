from __future__ import annotations

import pytest

from coola.utils.bloom_filter import BloomFilter

#################################
#     Tests for BloomFilter     #
#################################


@pytest.mark.parametrize("expected_items", [0, -1])
def test_bloom_filter_incorrect_expected_items(expected_items: int) -> None:
    with pytest.raises(ValueError, match="expected_items must be greater than 0"):
        BloomFilter(expected_items=expected_items, fp_rate=0.01)


@pytest.mark.parametrize("fp_rate", [0.0, 1.0, -0.1, 1.1])
def test_bloom_filter_incorrect_fp_rate(fp_rate: float) -> None:
    with pytest.raises(ValueError, match=r"fp_rate must be in \(0, 1\)"):
        BloomFilter(expected_items=100, fp_rate=fp_rate)


def test_bloom_filter_add_and_check_new_item_returns_false() -> None:
    bloom = BloomFilter(expected_items=100, fp_rate=0.01)
    assert bloom.add_and_check(b"hello") is False


def test_bloom_filter_add_and_check_repeated_item_returns_true() -> None:
    bloom = BloomFilter(expected_items=100, fp_rate=0.01)
    bloom.add_and_check(b"hello")
    assert bloom.add_and_check(b"hello") is True


def test_bloom_filter_never_has_false_negatives() -> None:
    bloom = BloomFilter(expected_items=1000, fp_rate=0.01)
    items = [f"item-{i}".encode() for i in range(1000)]
    for item in items:
        bloom.add_and_check(item)
    # Every item, once added, must always be reported as present.
    assert all(bloom.add_and_check(item) is True for item in items)


def test_bloom_filter_distinct_items_usually_not_flagged_as_duplicate() -> None:
    bloom = BloomFilter(expected_items=1000, fp_rate=0.001)
    items = [f"item-{i}".encode() for i in range(1000)]
    false_positives = sum(1 for item in items if bloom.add_and_check(item))
    # First-time additions should almost never be flagged as already
    # present, given the low configured fp_rate; allow generous slack
    # to avoid test flakiness while still catching a broken filter.
    assert false_positives < len(items) * 0.05


def test_bloom_filter_size_scales_with_expected_items() -> None:
    small = BloomFilter(expected_items=10, fp_rate=0.01)
    large = BloomFilter(expected_items=1_000_000, fp_rate=0.01)
    assert large.size > small.size


def test_bloom_filter_size_scales_with_fp_rate() -> None:
    loose = BloomFilter(expected_items=1000, fp_rate=0.5)
    strict = BloomFilter(expected_items=1000, fp_rate=0.0001)
    assert strict.size > loose.size
