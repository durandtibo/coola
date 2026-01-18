from __future__ import annotations

import pytest

from coola.testing import assert_allclose, assert_equal

#####################################
#     Tests for assert_equal     #
#####################################


def test_assert_equal_true_int() -> None:
    assert_equal(1, 1)


def test_assert_equal_true_list() -> None:
    assert_equal([1, 2, 3], [1, 2, 3])


def test_assert_equal_true_dict() -> None:
    assert_equal({"a": 1, "b": 2}, {"a": 1, "b": 2})


def test_assert_equal_true_nested() -> None:
    assert_equal(
        {"a": [1, 2, 3], "b": {"c": 4}},
        {"a": [1, 2, 3], "b": {"c": 4}},
    )


def test_assert_equal_false_different_values() -> None:
    with pytest.raises(AssertionError, match="Objects are not equal"):
        assert_equal([1, 2, 3], [1, 2, 4])


def test_assert_equal_false_different_types() -> None:
    with pytest.raises(AssertionError, match="Objects are not equal"):
        assert_equal([1, 2, 3], (1, 2, 3))


def test_assert_equal_false_different_length() -> None:
    with pytest.raises(AssertionError, match="Objects are not equal"):
        assert_equal([1, 2], [1, 2, 3])


def test_assert_equal_nan_false() -> None:
    with pytest.raises(AssertionError, match="Objects are not equal"):
        assert_equal(float("nan"), float("nan"))


def test_assert_equal_nan_true() -> None:
    assert_equal(float("nan"), float("nan"), equal_nan=True)


#########################################
#     Tests for assert_allclose     #
#########################################


def test_assert_allclose_true_exact() -> None:
    assert_allclose(1.0, 1.0)


def test_assert_allclose_true_close() -> None:
    assert_allclose(1.0, 1.0 + 1e-9)


def test_assert_allclose_true_list() -> None:
    assert_allclose([1.0, 2.0, 3.0], [1.0 + 1e-9, 2.0, 3.0 - 1e-9])


def test_assert_allclose_true_dict() -> None:
    assert_allclose(
        {"a": 1.0, "b": 2.0},
        {"a": 1.0 + 1e-9, "b": 2.0},
    )


def test_assert_allclose_true_nested() -> None:
    assert_allclose(
        {"a": [1.0, 2.0], "b": {"c": 3.0}},
        {"a": [1.0 + 1e-9, 2.0], "b": {"c": 3.0 - 1e-9}},
    )


def test_assert_allclose_true_atol() -> None:
    assert_allclose(1.0, 1.5, atol=0.5, rtol=0.0)


def test_assert_allclose_true_rtol() -> None:
    assert_allclose(1.0, 1.05, rtol=0.1, atol=0.0)


def test_assert_allclose_false_different_values() -> None:
    with pytest.raises(AssertionError, match="Objects are not approximately equal"):
        assert_allclose([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])


def test_assert_allclose_false_different_types() -> None:
    with pytest.raises(AssertionError, match="Objects are not approximately equal"):
        assert_allclose([1.0, 2.0, 3.0], (1.0, 2.0, 3.0))


def test_assert_allclose_false_outside_tolerance() -> None:
    with pytest.raises(AssertionError, match="Objects are not approximately equal"):
        assert_allclose(1.0, 1.1, atol=1e-8, rtol=1e-5)


def test_assert_allclose_nan_false() -> None:
    with pytest.raises(AssertionError, match="Objects are not approximately equal"):
        assert_allclose(float("nan"), float("nan"))


def test_assert_allclose_nan_true() -> None:
    assert_allclose(float("nan"), float("nan"), equal_nan=True)


def test_assert_allclose_negative_rtol() -> None:
    with pytest.raises(ValueError, match="rtol must be non-negative"):
        assert_allclose(1.0, 1.0, rtol=-0.1)


def test_assert_allclose_negative_atol() -> None:
    with pytest.raises(ValueError, match="atol must be non-negative"):
        assert_allclose(1.0, 1.0, atol=-0.1)
