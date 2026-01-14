from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig


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
