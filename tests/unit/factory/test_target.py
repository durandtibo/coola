from __future__ import annotations

import math
from collections import Counter
from typing import Any, Union

import pytest

from coola.factory.target import builds


class Animal:
    pass


class Dog(Animal):
    pass


class Cat(Animal):
    pass


def make_dog() -> Dog:
    return Dog()


def make_dog_or_none() -> Dog | None:
    return Dog()


def make_dog_or_cat() -> Dog | Cat:
    return Dog()


def make_union() -> Union[Dog, Cat]:  # noqa: UP007
    return Dog()


def make_list() -> list[int]:
    return [1]


def make_any() -> Any:
    return 1


def make_dog_or_any() -> Dog | Any:
    return Dog()


def make_without_hint():  # noqa: ANN201
    return Dog()


def make_unresolvable() -> UndefinedName:  # noqa: F821  # ty: ignore[unresolved-reference]
    return Dog()


def make_literal() -> 1:  # ty: ignore[invalid-type-form]
    return 1


############################
#     Tests for builds     #
############################


@pytest.mark.parametrize(
    ("target", "cls", "expected"),
    [
        pytest.param(Dog, Animal, True, id="subclass"),
        pytest.param(Dog, Dog, True, id="same class"),
        pytest.param(Animal, Dog, False, id="parent class"),
        pytest.param(Dog, Counter, False, id="unrelated class"),
        pytest.param(Dog, (Counter, Animal), True, id="tuple with a match"),
        pytest.param(Dog, (Counter, list), False, id="tuple without a match"),
        pytest.param(Counter, dict, True, id="builtin subclass"),
        pytest.param(make_dog, Animal, True, id="function with a class hint"),
        pytest.param(make_dog, Cat, False, id="function with a wrong class hint"),
        pytest.param(make_dog_or_none, Animal, True, id="A | None with a match"),
        pytest.param(make_dog_or_none, Cat, False, id="A | None without a match"),
        pytest.param(make_dog_or_cat, Cat, True, id="A | B"),
        pytest.param(make_dog_or_cat, Counter, False, id="A | B without a match"),
        pytest.param(make_union, Cat, True, id="Union[A, B]"),
        pytest.param(make_list, list, True, id="generic"),
        pytest.param(make_list, dict, False, id="generic without a match"),
        pytest.param(make_without_hint, Animal, None, id="function without hint"),
        pytest.param(make_unresolvable, Animal, None, id="unresolvable hint"),
        pytest.param(make_any, Animal, None, id="Any"),
        pytest.param(make_dog_or_any, Animal, None, id="union with Any"),
        pytest.param(make_literal, int, None, id="hint that is not a class"),
        pytest.param(math.isclose, float, None, id="builtin function"),
        pytest.param(math.pi, float, None, id="object that is not callable"),
        pytest.param(None, object, None, id="None"),
    ],
)
def test_builds(target: Any, cls: Any, expected: bool | None) -> None:
    assert builds(target, cls) is expected
