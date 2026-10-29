r"""Property-based tests for the guarantees of the registries."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import pytest
from hypothesis import assume, given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

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
    expected = model_resolve(registered, cls)
    assume(expected is not None)
    assert TypeRegistry[int](registered).resolve(cls) == expected


@given(registered=st.dictionaries(classes, values), cls=classes)
def test_resolve_raises_if_no_class_of_the_mro_is_registered(
    registered: dict[type, int], cls: type
) -> None:
    assume(model_resolve(registered, cls) is None)
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        TypeRegistry[int](registered).resolve(cls)


@given(registered=st.dictionaries(classes, values), cls=classes)
def test_resolve_prefers_the_class_itself(registered: dict[type, int], cls: type) -> None:
    registry = TypeRegistry[int]({**registered, cls: 1000})
    assert registry.resolve(cls) == 1000


@given(registered=st.dictionaries(classes, values, min_size=1), cls=classes)
def test_resolve_is_stable(registered: dict[type, int], cls: type) -> None:
    assume(model_resolve(registered, cls) is not None)
    registry = TypeRegistry[int](registered)
    first = registry.resolve(cls)
    assert registry.resolve(cls) == first


@given(registered=st.dictionaries(classes, values), cls=classes)
def test_resolve_error_is_stable(registered: dict[type, int], cls: type) -> None:
    assume(model_resolve(registered, cls) is None)
    registry = TypeRegistry[int](registered)
    with contextlib.suppress(TypeNotRegisteredError):
        registry.resolve(cls)
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        registry.resolve(cls)


@given(registered=st.dictionaries(classes, values), cls=classes)
def test_resolve_error_is_a_key_error_that_names_the_type(
    registered: dict[type, int], cls: type
) -> None:
    assume(model_resolve(registered, cls) is None)
    with pytest.raises(KeyError, match=cls.__name__):
        TypeRegistry[int](registered).resolve(cls)


@given(registered=st.dictionaries(classes, values))
def test_resolve_does_not_use_the_virtual_subclasses(registered: dict[type, int]) -> None:
    # ``str`` is a virtual subclass of ``Sequence``: it is only resolved with its real MRO.
    registry = TypeRegistry[int]({**registered, Sequence: -1})
    expected = model_resolve(registered, str)
    assume(expected is not None)
    assert registry.resolve(str) == expected


@given(registered=st.dictionaries(classes, values))
def test_resolve_raises_for_the_virtual_subclasses(registered: dict[type, int]) -> None:
    registry = TypeRegistry[int]({**registered, Sequence: -1})
    assume(model_resolve(registered, str) is None)
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        registry.resolve(str)


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

    def new_key(self, data: st.DataObject) -> str:
        r"""Draw a key that is not registered."""
        return data.draw(keys.filter(lambda key: key not in self.model))

    def registered_key(self, data: st.DataObject) -> str:
        r"""Draw a key that is registered."""
        return data.draw(st.sampled_from(sorted(self.model)))

    @rule(data=st.data(), value=values)
    def register(self, data: st.DataObject, value: int) -> None:
        key = self.new_key(data)
        self.registry.register(key, value)
        self.model[key] = value

    @precondition(lambda self: bool(self.model))
    @rule(data=st.data(), value=values)
    def register_duplicate(self, data: st.DataObject, value: int) -> None:
        with pytest.raises(AlreadyRegisteredError, match="already registered"):
            self.registry.register(self.registered_key(data), value)

    @rule(key=keys, value=values)
    def register_exist_ok(self, key: str, value: int) -> None:
        self.registry.register(key, value, exist_ok=True)
        self.model[key] = value

    @rule(data=st.data())
    def register_many(self, data: st.DataObject) -> None:
        items = data.draw(
            st.dictionaries(keys.filter(lambda k: k not in self.model), values, max_size=3)
        )
        self.registry.register_many(items)
        self.model.update(items)

    @rule(items=st.dictionaries(keys, values, max_size=3))
    def register_many_exist_ok(self, items: dict[str, int]) -> None:
        self.registry.register_many(items, exist_ok=True)
        self.model.update(items)

    @precondition(lambda self: bool(self.model))
    @rule(data=st.data(), value=values)
    def register_many_duplicate(self, data: st.DataObject, value: int) -> None:
        with pytest.raises(AlreadyRegisteredError, match="already registered"):
            self.registry.register_many({self.registered_key(data): value})

    @precondition(lambda self: bool(self.model))
    @rule(data=st.data())
    def unregister(self, data: st.DataObject) -> None:
        key = self.registered_key(data)
        assert self.registry.unregister(key) == self.model.pop(key)

    @rule(data=st.data())
    def unregister_missing(self, data: st.DataObject) -> None:
        with pytest.raises(KeyNotRegisteredError, match="is not registered"):
            self.registry.unregister(self.new_key(data))

    @rule(key=keys)
    def get(self, key: str) -> None:
        assert self.registry.get(key) == self.model.get(key)

    @rule(key=keys)
    def has(self, key: str) -> None:
        assert self.registry.has(key) == (key in self.model)

    @rule(key=keys)
    def contains(self, key: str) -> None:
        assert (key in self.registry) == (key in self.model)

    @precondition(lambda self: bool(self.model))
    @rule(data=st.data())
    def getitem(self, data: st.DataObject) -> None:
        key = self.registered_key(data)
        assert self.registry[key] == self.model[key]

    @rule(data=st.data())
    def getitem_missing(self, data: st.DataObject) -> None:
        with pytest.raises(KeyNotRegisteredError, match="is not registered"):
            self.registry[self.new_key(data)]

    @rule()
    def clear(self) -> None:
        self.registry.clear()
        self.model.clear()

    @invariant()
    def same_length(self) -> None:
        assert len(self.registry) == len(self.model)

    @invariant()
    def same_items(self) -> None:
        assert dict(self.registry.items()) == self.model

    @invariant()
    def same_keys(self) -> None:
        assert set(self.registry.keys()) == set(self.model)

    @invariant()
    def same_values(self) -> None:
        assert sorted(self.registry.values()) == sorted(self.model.values())

    @invariant()
    def same_iteration(self) -> None:
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

    def resolvable_classes(self) -> list[type]:
        r"""Return the classes that can be resolved."""
        return [cls for cls in CLASSES if model_resolve(self.model, cls) is not None]

    def unresolvable_classes(self) -> list[type]:
        r"""Return the classes that cannot be resolved."""
        return [cls for cls in CLASSES if model_resolve(self.model, cls) is None]

    @rule(cls=classes, value=values)
    def register(self, cls: type, value: int) -> None:
        self.registry.register(cls, value, exist_ok=True)
        self.model[cls] = value

    @precondition(lambda self: bool(self.model))
    @rule(data=st.data())
    def unregister(self, data: st.DataObject) -> None:
        cls = data.draw(st.sampled_from(list(self.model)))
        assert self.registry.unregister(cls) == self.model.pop(cls)

    @precondition(lambda self: len(self.model) < len(CLASSES))
    @rule(data=st.data())
    def unregister_missing(self, data: st.DataObject) -> None:
        cls = data.draw(st.sampled_from([c for c in CLASSES if c not in self.model]))
        with pytest.raises(TypeNotRegisteredError, match="is not registered"):
            self.registry.unregister(cls)

    @rule(items=st.dictionaries(classes, values, max_size=3))
    def register_many(self, items: dict[type, int]) -> None:
        self.registry.register_many(items, exist_ok=True)
        self.model.update(items)

    @rule()
    def clear(self) -> None:
        self.registry.clear()
        self.model.clear()

    @precondition(lambda self: bool(self.model))
    @rule(data=st.data())
    def resolve(self, data: st.DataObject) -> None:
        cls = data.draw(st.sampled_from(self.resolvable_classes()))
        assert self.registry.resolve(cls) == model_resolve(self.model, cls)

    @precondition(lambda self: bool(self.unresolvable_classes()))
    @rule(data=st.data())
    def resolve_missing(self, data: st.DataObject) -> None:
        cls = data.draw(st.sampled_from(self.unresolvable_classes()))
        with pytest.raises(TypeNotRegisteredError, match="is not registered"):
            self.registry.resolve(cls)

    @invariant()
    def resolve_every_resolvable_class(self) -> None:
        # Also the classes that were resolved before the last change (the cache).
        resolved = {cls: self.registry.resolve(cls) for cls in self.resolvable_classes()}
        assert resolved == {
            cls: model_resolve(self.model, cls) for cls in self.resolvable_classes()
        }

    @invariant()
    def resolve_error_for_every_unresolvable_class(self) -> None:
        for cls in self.unresolvable_classes():
            with pytest.raises(TypeNotRegisteredError, match="is not registered"):
                self.registry.resolve(cls)


TestRegistryMachine = RegistryMachine.TestCase
TestRegistryMachine.settings = settings(max_examples=50, stateful_step_count=30, deadline=None)
TestTypeRegistryMachine = TypeRegistryMachine.TestCase
TestTypeRegistryMachine.settings = settings(max_examples=50, stateful_step_count=30, deadline=None)
