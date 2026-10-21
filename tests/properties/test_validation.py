r"""Property-based tests for the type validators of ``coola.validation``."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, strategies as st

from coola.validation import (
    validate_bool,
    validate_dict,
    validate_float,
    validate_frozenset,
    validate_int,
    validate_isinstance,
    validate_list,
    validate_set,
    validate_str,
    validate_tuple,
)

VALIDATORS: dict[type, Any] = {
    bool: validate_bool,
    int: validate_int,
    float: validate_float,
    str: validate_str,
    list: validate_list,
    tuple: validate_tuple,
    dict: validate_dict,
    set: validate_set,
    frozenset: validate_frozenset,
}

samples = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(),
    st.text(),
    st.binary(),
    st.lists(st.integers(), max_size=3),
    st.tuples(st.integers()),
    st.dictionaries(st.text(max_size=2), st.integers(), max_size=3),
    st.sets(st.integers(), max_size=3),
    st.frozensets(st.integers(), max_size=3),
)


def _accepts(cls: type, value: object) -> bool:
    # ``bool`` is a subclass of ``int`` but ``validate_int`` rejects it.
    if cls is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, cls)


def _raises(validator: Any, value: object, **kwargs: Any) -> bool:
    try:
        validator(value, **kwargs)
    except TypeError:
        return True
    return False


@pytest.mark.parametrize("cls", list(VALIDATORS))
@given(value=samples)
def test_validator_accepts_exactly_the_expected_type(cls: type, value: object) -> None:
    assert _raises(VALIDATORS[cls], value) == (not _accepts(cls, value))


@pytest.mark.parametrize("cls", list(VALIDATORS))
@given(value=samples, name=st.text(alphabet="abcxyz_", min_size=1, max_size=8))
def test_validator_error_message_contains_the_name(cls: type, value: object, name: str) -> None:
    if not _accepts(cls, value):
        with pytest.raises(TypeError, match=f"^{name} must be"):
            VALIDATORS[cls](value, name=name)


@given(value=samples)
def test_validate_int_rejects_bool_but_validate_isinstance_does_not(value: object) -> None:
    if isinstance(value, bool):
        assert _raises(validate_int, value)
        assert not _raises(validate_isinstance, value, cls=int)


@given(value=samples)
def test_at_most_one_builtin_validator_accepts_a_value(value: object) -> None:
    accepted = [cls for cls in VALIDATORS if _accepts(cls, value)]
    assert len(accepted) <= 1


@given(items=st.lists(st.integers(), max_size=5))
def test_container_validators_do_not_convert_between_types(items: list[int]) -> None:
    assert not _raises(validate_list, items)
    for converted, validator in (
        (tuple(items), validate_list),
        (set(items), validate_list),
        (items, validate_tuple),
        (items, validate_set),
        (set(items), validate_frozenset),
        (frozenset(items), validate_set),
    ):
        assert _raises(validator, converted)
