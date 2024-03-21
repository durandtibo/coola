from __future__ import annotations

import random

from coola.random import RandomRandomSeedSetter

############################################
#     Tests for RandomRandomSeedSetter     #
############################################


def test_random_random_seed_setter_repr() -> None:
    assert repr(RandomRandomSeedSetter()).startswith("RandomRandomSeedSetter(")


def test_random_random_seed_setter_str() -> None:
    assert str(RandomRandomSeedSetter()).startswith("RandomRandomSeedSetter(")


def test_random_random_seed_setter_manual_seed() -> None:
    seed_setter = RandomRandomSeedSetter()
    seed_setter.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    seed_setter.manual_seed(42)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2
