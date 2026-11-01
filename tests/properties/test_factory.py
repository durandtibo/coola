r"""Property-based tests of ``coola.factory``: the factory of a builtin
type equals the direct call, an instance is returned as is, and the
configuration is not modified."""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import given, strategies as st

from coola.factory import OBJECT_TARGET, factory, resolve_object

# A key that is not reserved (`_target_`, `_init_`, `_allowed_prefixes_` start with `_`).
keys = st.text(min_size=1, max_size=5).filter(lambda key: not key.startswith("_"))
kwargs_strategy = st.dictionaries(keys, st.integers(), max_size=5)
types = st.sampled_from([list, dict, set, frozenset, tuple, int, str])
values = st.one_of(
    st.lists(st.integers(), max_size=5),
    st.dictionaries(keys, st.integers(), max_size=5),
    st.sets(st.integers(), max_size=5),
    st.frozensets(st.integers(), max_size=5),
    st.tuples(st.integers(), st.integers()),
    st.integers(),
    st.text(max_size=5),
)


@given(cls=types)
def test_resolve_object_without_arguments_equals_the_direct_call(cls: type) -> None:
    assert resolve_object({OBJECT_TARGET: f"builtins.{cls.__name__}"}, cls) == cls()


@given(kwargs=kwargs_strategy)
def test_resolve_object_dict_equals_the_direct_call(kwargs: dict[str, int]) -> None:
    assert resolve_object({OBJECT_TARGET: "builtins.dict", **kwargs}, dict) == dict(**kwargs)


@given(kwargs=kwargs_strategy)
def test_factory_dict_equals_the_direct_call(kwargs: dict[str, Any]) -> None:
    assert factory("builtins.dict", **kwargs) == dict(**kwargs)


@given(value=values)
def test_factory_with_a_positional_argument_equals_the_direct_call(value: Any) -> None:
    cls = type(value)
    assert factory(f"builtins.{cls.__name__}", value) == cls(value)


@given(value=values)
def test_resolve_object_returns_an_instance_as_is(value: Any) -> None:
    assert resolve_object(value, type(value)) is value


@given(kwargs=kwargs_strategy)
def test_resolve_object_does_not_modify_the_configuration(kwargs: dict[str, int]) -> None:
    config = {OBJECT_TARGET: "builtins.dict", **kwargs}
    expected = copy.deepcopy(config)
    resolve_object(config, dict)
    assert config == expected
