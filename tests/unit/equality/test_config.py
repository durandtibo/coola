from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig

####################################
#     Tests for EqualityConfig     #
####################################


def test_equality_config_default() -> None:
    config = EqualityConfig()
    assert config.equal_nan is False
    assert config.atol == 0.0
    assert config.rtol == 0.0
    assert config.show_difference is False


def test_equality_config_with_parameters() -> None:
    config = EqualityConfig(equal_nan=True, atol=1e-6, rtol=1e-5, show_difference=True)
    assert config.equal_nan is True
    assert config.atol == 1e-6
    assert config.rtol == 1e-5
    assert config.show_difference is True


def test_equality_config_negative_atol() -> None:
    with pytest.raises(ValueError, match="atol must be non-negative"):
        EqualityConfig(atol=-1.0)


def test_equality_config_negative_rtol() -> None:
    with pytest.raises(ValueError, match="rtol must be non-negative"):
        EqualityConfig(rtol=-0.5)


def test_equality_config_zero_tolerances() -> None:
    config = EqualityConfig(atol=0.0, rtol=0.0)
    assert config.atol == 0.0
    assert config.rtol == 0.0


def test_equality_config_max_depth_default() -> None:
    config = EqualityConfig()
    assert config.max_depth == 1000


def test_equality_config_max_depth_custom() -> None:
    config = EqualityConfig(max_depth=100)
    assert config.max_depth == 100


def test_equality_config_max_depth_zero() -> None:
    with pytest.raises(ValueError, match="max_depth must be positive"):
        EqualityConfig(max_depth=0)


def test_equality_config_max_depth_negative() -> None:
    with pytest.raises(ValueError, match="max_depth must be positive"):
        EqualityConfig(max_depth=-1)


def test_equality_config_depth() -> None:
    config = EqualityConfig()
    assert config.depth == 0


def test_equality_config_increment_depth() -> None:
    config = EqualityConfig()
    config.increment_depth()
    assert config.depth == 1
    config.increment_depth()
    assert config.depth == 2


def test_equality_config_decrement_depth() -> None:
    config = EqualityConfig()
    config.decrement_depth()
    assert config.depth == -1
    config.decrement_depth()
    assert config.depth == -2


def test_equality_config_eq_true() -> None:
    assert EqualityConfig() == EqualityConfig()


def test_equality_config_eq_true_with_parameters() -> None:
    assert EqualityConfig(
        equal_nan=True, atol=1e-6, rtol=1e-5, show_difference=True
    ) == EqualityConfig(equal_nan=True, atol=1e-6, rtol=1e-5, show_difference=True)


def test_equality_config_eq_true_different_depth() -> None:
    config = EqualityConfig()
    config._current_depth = 42
    assert config == EqualityConfig()


def test_equality_config_eq_false_different_equal_nan() -> None:
    assert EqualityConfig() != EqualityConfig(equal_nan=True)


def test_equality_config_eq_false_different_atol() -> None:
    assert EqualityConfig() != EqualityConfig(atol=1e-6)


def test_equality_config_eq_false_different_rtol() -> None:
    assert EqualityConfig() != EqualityConfig(rtol=1e-6)


def test_equality_config_eq_false_different_show_difference() -> None:
    assert EqualityConfig() != EqualityConfig(show_difference=True)


def test_equality_config_eq_false_different_max_depth() -> None:
    assert EqualityConfig() != EqualityConfig(max_depth=100)


def test_equality_config_large_tolerance_warning() -> None:
    with pytest.warns(UserWarning, match="Very large tolerance values detected"):
        EqualityConfig(atol=1e11)


def test_equality_config_large_rtol_warning() -> None:
    with pytest.warns(UserWarning, match="Very large tolerance values detected"):
        EqualityConfig(rtol=1e15)


def test_equality_config_is_visited_false() -> None:
    config = EqualityConfig()
    obj1 = {"key": "value"}
    obj2 = {"key": "value"}
    assert not config.is_visited(obj1, obj2)


def test_equality_config_mark_visited() -> None:
    config = EqualityConfig()
    obj1 = {"key": "value"}
    obj2 = {"key": "value"}
    config.mark_visited(obj1, obj2)
    assert config.is_visited(obj1, obj2)


def test_equality_config_unmark_visited() -> None:
    config = EqualityConfig()
    obj1 = {"key": "value"}
    obj2 = {"key": "value"}
    config.mark_visited(obj1, obj2)
    assert config.is_visited(obj1, obj2)
    config.unmark_visited(obj1, obj2)
    assert not config.is_visited(obj1, obj2)


def test_equality_config_visited_different_pairs() -> None:
    config = EqualityConfig()
    obj1 = {"key": "value1"}
    obj2 = {"key": "value2"}
    obj3 = {"key": "value3"}
    config.mark_visited(obj1, obj2)
    assert config.is_visited(obj1, obj2)
    assert not config.is_visited(obj1, obj3)
    assert not config.is_visited(obj2, obj3)
