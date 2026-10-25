from __future__ import annotations

import contextlib

import pytest

from coola.equality.config import EqualityConfig

####################################
#     Tests for EqualityConfig     #
####################################


def test_equality_config_default_equal_nan() -> None:
    assert EqualityConfig().equal_nan is False


def test_equality_config_default_atol() -> None:
    assert EqualityConfig().atol == 0.0


def test_equality_config_default_rtol() -> None:
    assert EqualityConfig().rtol == 0.0


def test_equality_config_default_reports() -> None:
    assert EqualityConfig().reports is None


def test_equality_config_default_is_not_explaining() -> None:
    assert not EqualityConfig().explaining


def test_equality_config_with_parameters_equal_nan() -> None:
    assert EqualityConfig(equal_nan=True).equal_nan is True


def test_equality_config_with_parameters_atol() -> None:
    assert EqualityConfig(atol=1e-6).atol == 1e-6


def test_equality_config_with_parameters_rtol() -> None:
    assert EqualityConfig(rtol=1e-5).rtol == 1e-5


def test_equality_config_with_reports_is_explaining() -> None:
    assert EqualityConfig(reports=[]).explaining


def test_equality_config_negative_atol() -> None:
    with pytest.raises(ValueError, match="atol must be non-negative"):
        EqualityConfig(atol=-1.0)


def test_equality_config_negative_rtol() -> None:
    with pytest.raises(ValueError, match="rtol must be non-negative"):
        EqualityConfig(rtol=-0.5)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
@pytest.mark.parametrize("name", ["atol", "rtol"])
def test_equality_config_non_finite_tolerance(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{name} must be"):
        EqualityConfig(**{name: value})


def test_equality_config_unknown_option() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        EqualityConfig(unknown=1)


def test_equality_config_zero_atol() -> None:
    assert EqualityConfig(atol=0.0, rtol=0.0).atol == 0.0


def test_equality_config_zero_rtol() -> None:
    assert EqualityConfig(atol=0.0, rtol=0.0).rtol == 0.0


def test_equality_config_max_depth_default() -> None:
    config = EqualityConfig()
    assert config.max_depth == 100


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


def test_equality_config_increment_depth_twice() -> None:
    config = EqualityConfig()
    config.increment_depth()
    config.increment_depth()
    assert config.depth == 2


def test_equality_config_decrement_depth() -> None:
    config = EqualityConfig()
    config.decrement_depth()
    assert config.depth == -1


def test_equality_config_decrement_depth_twice() -> None:
    config = EqualityConfig()
    config.decrement_depth()
    config.decrement_depth()
    assert config.depth == -2


def test_equality_config_eq_true() -> None:
    assert EqualityConfig() == EqualityConfig()


def test_equality_config_eq_true_with_parameters() -> None:
    assert EqualityConfig(equal_nan=True, atol=1e-6, rtol=1e-5, reports=[]) == EqualityConfig(
        equal_nan=True, atol=1e-6, rtol=1e-5, reports=[]
    )


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


def test_equality_config_eq_true_different_reports() -> None:
    assert EqualityConfig() == EqualityConfig(reports=[])


def test_equality_config_eq_false_different_max_depth() -> None:
    assert EqualityConfig() != EqualityConfig(max_depth=50)


#################################
#     Tests for mark_warned     #
#################################


def test_equality_config_mark_warned_first_time() -> None:
    assert EqualityConfig().mark_warned("atol") is True


def test_equality_config_mark_warned_second_time() -> None:
    config = EqualityConfig()
    config.mark_warned("atol")
    assert config.mark_warned("atol") is False


def test_equality_config_mark_warned_other_name() -> None:
    config = EqualityConfig()
    config.mark_warned("atol")
    assert config.mark_warned("rtol") is True


def test_equality_config_mark_warned_is_per_config() -> None:
    EqualityConfig().mark_warned("atol")
    assert EqualityConfig().mark_warned("atol") is True


##############################
#     Tests for visiting     #
##############################


def test_equality_config_visiting_first_time() -> None:
    with EqualityConfig().visiting([], []) as in_progress:
        assert in_progress is False


def test_equality_config_visiting_same_pair_is_in_progress() -> None:
    config = EqualityConfig()
    a, b = [], []
    with config.visiting(a, b), config.visiting(a, b) as inner:
        assert inner is True


def test_equality_config_visiting_swapped_pair_is_not_in_progress() -> None:
    config = EqualityConfig()
    a, b = [], []
    with config.visiting(a, b), config.visiting(b, a) as swapped:
        assert swapped is False


def test_equality_config_visiting_pair_is_removed_on_exit() -> None:
    config = EqualityConfig()
    a, b = [], []
    with config.visiting(a, b):
        pass
    with config.visiting(a, b) as in_progress:
        assert in_progress is False


def _fail() -> None:
    msg = "boom"
    raise RuntimeError(msg)


def test_equality_config_visiting_pair_is_removed_on_error() -> None:
    config = EqualityConfig()
    a, b = [], []
    with contextlib.suppress(RuntimeError), config.visiting(a, b):
        _fail()
    with config.visiting(a, b) as in_progress:
        assert in_progress is False


@pytest.mark.parametrize("name", ["atol", "rtol"])
@pytest.mark.parametrize("value", [True, False, "a", None])
def test_equality_config_tolerance_wrong_type(name: str, value: object) -> None:
    with pytest.raises(TypeError, match=f"{name} must be a real number"):
        EqualityConfig(**{name: value})


@pytest.mark.parametrize("value", [True, 1.5, "a", None])
def test_equality_config_max_depth_wrong_type(value: object) -> None:
    with pytest.raises(TypeError, match="max_depth must be an int"):
        EqualityConfig(max_depth=value)


@pytest.mark.parametrize("name", ["atol", "rtol"])
def test_equality_config_tolerance_int(name: str) -> None:
    assert getattr(EqualityConfig(**{name: 1}), name) == 1
