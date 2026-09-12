from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from coola.iterator.bootstrap import register_default_handlers


class _FakeRegistry:
    """Minimal stand-in for ``ChildFinderRegistry``/``IteratorRegistry``
    that only records what was passed to ``register_many``."""

    def __init__(self) -> None:
        self.state: dict[type, Any] = {}

    def register_many(self, mapping: Mapping[type, Any], exist_ok: bool = False) -> None:
        if not exist_ok:
            duplicates = set(mapping).intersection(self.state)
            if duplicates:
                msg = f"already registered: {duplicates}"
                raise RuntimeError(msg)
        self.state.update(mapping)


##############################################
#     Tests for register_default_handlers     #
##############################################


@pytest.mark.parametrize("data_type", [object, str, bytes, int, float, complex, bool])
def test_register_default_handlers_scalar_types(data_type: type) -> None:
    registry = _FakeRegistry()
    default_handler = "default"
    register_default_handlers(registry, default_handler, "iterable", "mapping")
    assert registry.state[data_type] == default_handler


@pytest.mark.parametrize("data_type", [list, tuple, range, Iterable, set, frozenset])
def test_register_default_handlers_iterable_types(data_type: type) -> None:
    registry = _FakeRegistry()
    iterable_handler = "iterable"
    register_default_handlers(registry, "default", iterable_handler, "mapping")
    assert registry.state[data_type] == iterable_handler


@pytest.mark.parametrize("data_type", [dict, Mapping])
def test_register_default_handlers_mapping_types(data_type: type) -> None:
    registry = _FakeRegistry()
    mapping_handler = "mapping"
    register_default_handlers(registry, "default", "iterable", mapping_handler)
    assert registry.state[data_type] == mapping_handler


def test_register_default_handlers_registers_all_types_exactly_once() -> None:
    registry = _FakeRegistry()
    register_default_handlers(registry, "default", "iterable", "mapping")
    expected_types = {
        object,
        str,
        bytes,
        int,
        float,
        complex,
        bool,
        list,
        tuple,
        range,
        Iterable,
        set,
        frozenset,
        dict,
        Mapping,
    }
    assert set(registry.state) == expected_types


def test_register_default_handlers_calls_register_many_once() -> None:
    """The bootstrap helper should populate the registry with a single
    ``register_many`` call rather than one call per type."""
    calls: list[Mapping[type, Any]] = []

    class _RecordingRegistry(_FakeRegistry):
        def register_many(self, mapping: Mapping[type, Any], exist_ok: bool = False) -> None:
            calls.append(dict(mapping))
            super().register_many(mapping, exist_ok=exist_ok)

    registry = _RecordingRegistry()
    register_default_handlers(registry, "default", "iterable", "mapping")
    assert len(calls) == 1
