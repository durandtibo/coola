from __future__ import annotations

from collections import OrderedDict

import pytest

from coola.equality import compare, objects_are_allclose, objects_are_equal

COMPARE = [objects_are_equal, objects_are_allclose]


@pytest.mark.parametrize("compare_fn", COMPARE)
def test_ordered_dict_same_order_is_equal(compare_fn) -> None:  # noqa: ANN001
    assert compare_fn(OrderedDict(a=1, b=2), OrderedDict(a=1, b=2))


@pytest.mark.parametrize("compare_fn", COMPARE)
def test_ordered_dict_different_order_is_not_equal(compare_fn) -> None:  # noqa: ANN001
    assert not compare_fn(OrderedDict(a=1, b=2), OrderedDict(b=2, a=1))


@pytest.mark.parametrize("compare_fn", COMPARE)
def test_ordered_dict_and_dict_are_not_equal(compare_fn) -> None:  # noqa: ANN001
    assert not compare_fn(OrderedDict(a=1, b=2), {"a": 1, "b": 2})
    assert not compare_fn({"a": 1, "b": 2}, OrderedDict(a=1, b=2))


@pytest.mark.parametrize("compare_fn", COMPARE)
def test_dict_ignores_key_order(compare_fn) -> None:  # noqa: ANN001
    assert compare_fn({"a": 1, "b": 2}, {"b": 2, "a": 1})


@pytest.mark.parametrize("compare_fn", COMPARE)
def test_ordered_dict_nested_order_is_checked(compare_fn) -> None:  # noqa: ANN001
    assert compare_fn([OrderedDict(a=1, b=2)], [OrderedDict(a=1, b=2)])
    assert not compare_fn([OrderedDict(a=1, b=2)], [OrderedDict(b=2, a=1)])
    assert not compare_fn({"k": OrderedDict(a=1, b=2)}, {"k": OrderedDict(b=2, a=1)})


@pytest.mark.parametrize("compare_fn", COMPARE)
def test_ordered_dict_with_mixed_key_types(compare_fn) -> None:  # noqa: ANN001
    assert compare_fn(OrderedDict([(1, "a"), ("x", 2)]), OrderedDict([(1, "a"), ("x", 2)]))
    assert not compare_fn(OrderedDict([(1, "a"), ("x", 2)]), OrderedDict([("x", 2), (1, "a")]))


def test_ordered_dict_allclose_applies_tolerance_to_values() -> None:
    assert objects_are_allclose(OrderedDict(a=1.0, b=2.0), OrderedDict(a=1.001, b=2.001), atol=0.01)
    assert not objects_are_allclose(
        OrderedDict(a=1.0, b=2.0), OrderedDict(b=2.001, a=1.001), atol=0.01
    )


def test_ordered_dict_compare_explains_the_order_difference() -> None:
    result = compare(OrderedDict(a=1, b=2), OrderedDict(b=2, a=1))
    assert not result.equal
    message = result.format_message()
    assert "mappings have the same keys in a different order" in message
    assert "['a', 'b']" in message
    assert "['b', 'a']" in message
