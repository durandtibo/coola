r"""Property-based tests (Hypothesis) for ``TypeRegistry`` type
resolution."""

from __future__ import annotations

from hypothesis import given, strategies as st

from coola.registry import TypeNotRegisteredError, TypeRegistry

# A small class hierarchy with multiple inheritance, so that MRO lookups
# are non-trivial.


class A: ...


class B(A): ...


class C(A): ...


class D(B, C): ...


class E(D): ...


class F: ...


class G(F, E): ...


TYPES = [object, A, B, C, D, E, F, G, int, bool, str]
types = st.sampled_from(TYPES)
states = st.dictionaries(types, st.integers(), max_size=len(TYPES))


def expected_resolution(state: dict[type, int], dtype: type) -> int | None:
    r"""Reference implementation: first registered type in the MRO."""
    for base in dtype.__mro__:
        if base in state:
            return state[base]
    return None


def resolve_or_none(registry: TypeRegistry[int], dtype: type) -> int | None:
    try:
        return registry.resolve(dtype)
    except TypeNotRegisteredError:
        return None


@given(states, types)
def test_type_registry_resolve_follows_mro(state: dict[type, int], dtype: type) -> None:
    registry = TypeRegistry[int](state)
    assert resolve_or_none(registry, dtype) == expected_resolution(state, dtype)


@given(states, types)
def test_type_registry_resolve_is_idempotent(state: dict[type, int], dtype: type) -> None:
    registry = TypeRegistry[int](state)
    first = resolve_or_none(registry, dtype)
    assert resolve_or_none(registry, dtype) == first


@given(states, st.lists(st.tuples(types, st.integers()), max_size=10), types)
def test_type_registry_cache_is_invalidated_by_register(
    state: dict[type, int], updates: list[tuple[type, int]], dtype: type
) -> None:
    registry = TypeRegistry[int](state)
    expected = dict(state)
    for key, value in updates:
        resolve_or_none(registry, dtype)  # populate the cache
        registry.register(key, value, exist_ok=True)
        expected[key] = value
        assert resolve_or_none(registry, dtype) == expected_resolution(expected, dtype)


@given(states, types, types)
def test_type_registry_cache_is_invalidated_by_unregister(
    state: dict[type, int], to_remove: type, dtype: type
) -> None:
    registry = TypeRegistry[int](state)
    resolve_or_none(registry, dtype)  # populate the cache
    expected = dict(state)
    if to_remove in expected:
        registry.unregister(to_remove)
        del expected[to_remove]
    assert resolve_or_none(registry, dtype) == expected_resolution(expected, dtype)
