from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence

import pytest

from coola.summary import (
    DefaultSummarizer,
    MappingSummarizer,
    NDArraySummarizer,
    SequenceSummarizer,
    SetSummarizer,
    SummarizerRegistry,
    TensorSummarizer,
    get_default_registry,
    register_summarizers,
    summarize,
)
from coola.testing.fixtures import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

if is_torch_available():  # pragma: no cover
    import torch

if is_numpy_available():  # pragma: no cover
    import numpy as np


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class CustomList(list):
    r"""Create a custom class that inherits from list."""


###############################
#     Tests for summarize     #
###############################


def test_summarize_with_simple_value() -> None:
    assert summarize(5) == "<class 'int'> 5"


def test_summarize_with_string() -> None:
    assert summarize("hello") == "<class 'str'> hello"


def test_summarize_with_dict() -> None:
    assert summarize({"a": 1, "b": 2}) == "<class 'dict'> (length=2)\n  (a): 1\n  (b): 2"


def test_summarize_with_list() -> None:
    assert summarize([1, 2, 3]) == "<class 'list'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"


def test_summarize_with_max_depth_1() -> None:
    assert (
        summarize({"level1": {"level2": {"level3": [1, 2, 3]}}})
        == "<class 'dict'> (length=1)\n  (level1): {'level2': {'level3': [1, 2, 3]}}"
    )


def test_summarize_with_max_depth_2() -> None:
    assert (
        summarize({"level1": {"level2": {"level3": [1, 2, 3]}}}, max_depth=2)
        == "<class 'dict'> (length=1)\n"
        "  (level1): <class 'dict'> (length=1)\n"
        "      (level2): {'level3': [1, 2, 3]}"
    )


def test_summarize_with_max_depth_3() -> None:
    assert (
        summarize({"level1": {"level2": {"level3": [1, 2, 3]}}}, max_depth=3)
        == "<class 'dict'> (length=1)\n"
        "  (level1): <class 'dict'> (length=1)\n"
        "      (level2): <class 'dict'> (length=1)\n"
        "          (level3): [1, 2, 3]"
    )


def test_summarize_with_custom_registry() -> None:
    assert (
        summarize({"a": 1, "b": 2}, registry=SummarizerRegistry({object: DefaultSummarizer()}))
        == "<class 'dict'> {'a': 1, 'b': 2}"
    )


##########################################
#     Tests for register_summarizers     #
##########################################


def test_register_summarizers_calls_registry() -> None:
    register_summarizers({CustomList: SequenceSummarizer()})
    assert get_default_registry().has_summarizer(CustomList)


def test_register_summarizers_with_exist_ok_true() -> None:
    register_summarizers({CustomList: SetSummarizer()})
    register_summarizers({CustomList: SequenceSummarizer()}, exist_ok=True)


def test_register_summarizers_with_exist_ok_false() -> None:
    register_summarizers({CustomList: SetSummarizer()})
    with pytest.raises(RuntimeError, match="already registered"):
        register_summarizers({CustomList: SequenceSummarizer()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    """Test that get_default_registry returns a SummarizerRegistry
    instance."""
    registry = get_default_registry()
    assert isinstance(registry, SummarizerRegistry)


def test_get_default_registry_returns_singleton() -> None:
    """Test that get_default_registry returns the same instance on
    multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


@pytest.mark.parametrize("dtype", [object, int, float, str, bool, complex])
def test_get_default_registry_default(dtype: type) -> None:
    """Test that scalar types are registered with DefaultSummarizer."""
    registry = get_default_registry()
    assert registry.has_summarizer(dtype)
    assert isinstance(registry.find_summarizer(dtype), DefaultSummarizer)


@pytest.mark.parametrize("dtype", [list, tuple, Sequence])
def test_get_default_registry_sequences(dtype: type) -> None:
    """Test that sequence types are registered with
    SequenceSummarizer."""
    registry = get_default_registry()
    assert registry.has_summarizer(dtype)
    assert isinstance(registry.find_summarizer(dtype), SequenceSummarizer)


@pytest.mark.parametrize("dtype", [set, frozenset])
def test_get_default_registry_sets(dtype: type) -> None:
    """Test that set types are registered with SetSummarizer."""
    registry = get_default_registry()
    assert registry.has_summarizer(dtype)
    assert isinstance(registry.find_summarizer(dtype), SetSummarizer)


@pytest.mark.parametrize("dtype", [dict, Mapping])
def test_register_default_summarizers_registers_mappings(dtype: type) -> None:
    """Test that mapping types are registered with MappingSummarizer."""
    registry = get_default_registry()
    assert registry.has_summarizer(dtype)
    assert isinstance(registry.find_summarizer(dtype), MappingSummarizer)


@numpy_available
def test_register_default_summarizers_registers_ndarray() -> None:
    registry = get_default_registry()
    assert registry.has_summarizer(np.ndarray)
    assert isinstance(registry.find_summarizer(np.ndarray), NDArraySummarizer)


@torch_available
def test_register_default_summarizers_registers_tensor() -> None:
    registry = get_default_registry()
    assert registry.has_summarizer(torch.Tensor)
    assert isinstance(registry.find_summarizer(torch.Tensor), TensorSummarizer)


def test_default_registry_can_summarize_list() -> None:
    """Test that default registry can summarize a list."""
    registry = get_default_registry()
    assert (
        registry.summarize([1, 2, 3]) == "<class 'list'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"
    )


def test_default_registry_can_summarize_dict() -> None:
    """Test that default registry can summarize a dict."""
    registry = get_default_registry()
    assert registry.summarize({"a": 1, "b": 2}) == "<class 'dict'> (length=2)\n  (a): 1\n  (b): 2"


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist across calls."""
    registry1 = get_default_registry()
    assert not registry1.has_summarizer(CustomList)
    registry1.register(CustomList, SequenceSummarizer())
    assert registry1.has_summarizer(CustomList)

    # Get registry again
    registry2 = get_default_registry()
    assert registry1 is registry2
    assert registry2.has_summarizer(CustomList)
