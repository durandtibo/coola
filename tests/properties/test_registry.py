r"""Property-based tests for the guarantees of the registries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from coola.registry import (
    AlreadyRegisteredError,
    KeyNotRegisteredError,
    Registry,
    TypeNotRegisteredError,
    TypeRegistry,
)
from tests.properties.strategies import keys

# A hierarchy with a diamond: E -> D -> (B, C) -> A.
A = type("A", (), {})
B = type("B", (A,), {})
C = type("C", (A,), {})
D = type("D", (B, C), {})
E = type("E", (D,), {})
F = type("F", (), {})  # unrelated to the others
CLASSES = [A, B, C, D, E, F, object, int, bool, str]
classes = st.sampled_from(CLASSES)
values = st.integers(0, 100)


def model_resolve(model: dict[type, int], cls: type) -> int | None:
    r"""Resolve ``cls`` in a dictionary with the method resolution
    order."""
    for base in cls.__mro__:
        if base in model:
            return model[base]
    return None


###############################
#     TypeRegistry.resolve    #
###############################


@given(registered=st.dictionaries(classes, values), cls=classes)
def test_resolve_returns_the_value_of_the_first_registered_class_of_the_mro(
    registered: dict[type, int], cls: type
) -> None:
    registry = TypeRegistry[int](registered)
    expected = model_resolve(registered, cls)
    if expected is None:
        with pytest.raises(TypeNotRegisteredError):
            registry.resolve(cls)
    else:
        assert registry.resolve(cls) == expected


@given(registered=st.dictionaries(classes, values), cls=classes)
def test_resolve_prefers_the_class_itself(registered: dict[type, int], cls: type) -> None:
    registry = TypeRegistry[int]({**registered, cls: 1000})
    assert registry.resolve(cls) == 1000


@given(registered=st.dictionaries(classes, values, min_size=1), cls=classes)
def test_resolve_is_stable(registered: dict[type, int], cls: type) -> None:
    registry = TypeRegistry[int](registered)
    try:
        first = registry.resolve(cls)
    except TypeNotRegisteredError:
        with pytest.raises(TypeNotRegisteredError):
            registry.resolve(cls)
    else:
        assert registry.resolve(cls) == first


@given(registered=st.dictionaries(classes, values), cls=classes)
def test_resolve_error_is_a_key_error_that_names_the_type(
    registered: dict[type, int], cls: type
) -> None:
    registry = TypeRegistry[int](registered)
    if model_resolve(registered, cls) is None:
        with pytest.raises(KeyError, match=cls.__name__):
            registry.resolve(cls)


@given(registered=st.dictionaries(classes, values))
def test_resolve_does_not_use_the_virtual_subclasses(registered: dict[type, int]) -> None:
    # ``str`` is a virtual subclass of ``Sequence``: it is only resolved with its real MRO.
    registry = TypeRegistry[int]({**registered, Sequence: -1})
    expected = model_resolve(registered, str)
    if expected is None:
        with pytest.raises(TypeNotRegisteredError):
            registry.resolve(str)
    else:
        assert registry.resolve(str) == expected


#########################################
#     Registry/TypeRegistry vs a dict   #
#########################################


class RegistryMachine(RuleBasedStateMachine):
    r"""Compare a ``Registry`` with a dictionary after any sequence of
    operations."""

    def __init__(self) -> None:
        super().__init__()
        self.registry = Registry[str, int]()
        self.model: dict[str, int] = {}

    @rule(key=keys, value=values)
    def register(self, key: str, value: int) -> None:
        if key in self.model:
            with pytest.raises(AlreadyRegisteredError):
                self.registry.register(key, value)
        else:
            self.registry.register(key, value)
            self.model[key] = value

    @rule(key=keys, value=values)
    def register_exist_ok(self, key: str, value: int) -> None:
        self.registry.register(key, value, exist_ok=True)
        self.model[key] = value

    @rule(items=st.dictionaries(keys, values, max_size=3), exist_ok=st.booleans())
    def register_many(self, items: dict[str, int], exist_ok: bool) -> None:
        if not exist_ok and set(items) & set(self.model):
            with pytest.raises(AlreadyRegisteredError):
                self.registry.register_many(items)
        else:
            self.registry.register_many(items, exist_ok=exist_ok)
            self.model.update(items)

    @rule(key=keys)
    def unregister(self, key: str) -> None:
        if key in self.model:
            assert self.registry.unregister(key) == self.model.pop(key)
        else:
            with pytest.raises(KeyNotRegisteredError):
                self.registry.unregister(key)

    @rule(key=keys)
    def get(self, key: str) -> None:
        assert self.registry.get(key) == self.model.get(key)
        assert self.registry.has(key) == (key in self.model)
        assert (key in self.registry) == (key in self.model)
        if key in self.model:
            assert self.registry[key] == self.model[key]
        else:
            with pytest.raises(KeyNotRegisteredError):
                self.registry[key]

    @rule()
    def clear(self) -> None:
        self.registry.clear()
        self.model.clear()

    @invariant()
    def same_content(self) -> None:
        assert len(self.registry) == len(self.model)
        assert dict(self.registry.items()) == self.model
        assert set(self.registry.keys()) == set(self.model)
        assert sorted(self.registry.values()) == sorted(self.model.values())
        assert set(self.registry) == set(self.model)


class TypeRegistryMachine(RuleBasedStateMachine):
    r"""Compare the resolution of a ``TypeRegistry`` with a dictionary
    and the method resolution order, after any sequence of operations:

    the cache of ``resolve`` is invalidated by the changes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.registry = TypeRegistry[int]()
        self.model: dict[type, int] = {}

    @rule(cls=classes, value=values)
    def register(self, cls: type, value: int) -> None:
        self.registry.register(cls, value, exist_ok=True)
        self.model[cls] = value

    @rule(cls=classes)
    def unregister(self, cls: type) -> None:
        if cls in self.model:
            assert self.registry.unregister(cls) == self.model.pop(cls)
        else:
            with pytest.raises(TypeNotRegisteredError):
                self.registry.unregister(cls)

    @rule(items=st.dictionaries(classes, values, max_size=3))
    def register_many(self, items: dict[type, int]) -> None:
        self.registry.register_many(items, exist_ok=True)
        self.model.update(items)

    @rule()
    def clear(self) -> None:
        self.registry.clear()
        self.model.clear()

    @rule(cls=classes)
    def resolve(self, cls: type) -> None:
        self.check(cls)

    def check(self, cls: type) -> Any:
        expected = model_resolve(self.model, cls)
        if expected is None:
            with pytest.raises(TypeNotRegisteredError):
                self.registry.resolve(cls)
        else:
            assert self.registry.resolve(cls) == expected

    @invariant()
    def resolve_every_class(self) -> None:
        # Also the classes that were resolved before the last change (the cache).
        for cls in CLASSES:
            self.check(cls)


TestRegistryMachine = RegistryMachine.TestCase
TestRegistryMachine.settings = settings(max_examples=50, stateful_step_count=30, deadline=None)
TestTypeRegistryMachine = TypeRegistryMachine.TestCase
TestTypeRegistryMachine.settings = settings(max_examples=50, stateful_step_count=30, deadline=None)
