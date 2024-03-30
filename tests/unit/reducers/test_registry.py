import copy
from unittest.mock import Mock, patch

import pytest

from coola.reducers import NativeReducer, ReducerRegistry


@pytest.fixture(autouse=True)
def _reset() -> None:
    state = copy.deepcopy(ReducerRegistry.registry)
    try:
        yield
    finally:
        ReducerRegistry.registry = state


#####################################
#     Tests for ReducerRegistry     #
#####################################


def test_reducer_registry_str() -> None:
    assert str(ReducerRegistry()).startswith("ReducerRegistry(")


def test_reducer_registry_default() -> None:
    assert len(ReducerRegistry.registry) >= 1
    assert isinstance(ReducerRegistry.registry["native"], NativeReducer)


@patch.dict(ReducerRegistry.registry, {}, clear=True)
def test_summarizer_add_reducer() -> None:
    reducer = Mock(spec=NativeReducer)
    ReducerRegistry.add_reducer("my_reducer", reducer)
    assert ReducerRegistry.registry["my_reducer"] == reducer


@patch.dict(ReducerRegistry.registry, {}, clear=True)
def test_summarizer_add_reducer_duplicate_exist_ok_true() -> None:
    reducer = Mock(spec=NativeReducer)
    ReducerRegistry.add_reducer("my_reducer", Mock(spec=NativeReducer))
    ReducerRegistry.add_reducer("my_reducer", reducer, exist_ok=True)
    assert ReducerRegistry.registry["my_reducer"] == reducer


@patch.dict(ReducerRegistry.registry, {}, clear=True)
def test_summarizer_add_reducer_duplicate_exist_ok_false() -> None:
    reducer = Mock(spec=NativeReducer)
    ReducerRegistry.add_reducer("my_reducer", Mock(spec=NativeReducer))
    with pytest.raises(RuntimeError, match="A reducer (.*) is already registered"):
        ReducerRegistry.add_reducer("my_reducer", reducer)


def test_summarizer_available_reducers() -> None:
    assert "native" in ReducerRegistry.available_reducers()


def test_summarizer_has_reducer_true() -> None:
    assert ReducerRegistry.has_reducer("native")


def test_summarizer_has_reducer_false() -> None:
    assert not ReducerRegistry.has_reducer("missing")
