from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    JaxArraySummarizer,
    SummarizerRegistry,
)
from coola.testing.fixtures import jax_available
from coola.utils.imports import MissingPackageError, is_jax_available

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry()


SUMMARIZERS = [
    JaxArraySummarizer,
]


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_default(summarizer_cls: type) -> None:
    assert summarizer_cls()._show_data is False


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True)._show_data is True


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_default(summarizer_cls: type) -> None:
    assert repr(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_show_data(summarizer_cls: type) -> None:
    assert repr(summarizer_cls(show_data=True)) == f"{summarizer_cls.__name__}(show_data=True)"


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_str(summarizer_cls: type) -> None:
    assert str(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_default(summarizer_cls: type) -> None:
    assert summarizer_cls().equal(summarizer_cls())


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True).equal(summarizer_cls(show_data=True))


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_show_data(summarizer_cls: type) -> None:
    assert not summarizer_cls(show_data=False).equal(summarizer_cls(show_data=True))


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_type(summarizer_cls: type) -> None:
    assert not summarizer_cls().equal(42)


@jax_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_child_class(summarizer_cls: type) -> None:
    class Child(summarizer_cls): ...

    assert not summarizer_cls().equal(Child())


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.jax.is_jax_available", lambda: False),
        pytest.raises(MissingPackageError, match="'jax' package is required"),
    ):
        summarizer_cls()


@jax_available
def test_jax_array_summarizer_summarize(registry: SummarizerRegistry) -> None:
    result = JaxArraySummarizer().summarize(jnp.ones((2, 3)), registry)
    assert result.endswith(" | shape=(2, 3) | dtype=float32")


@jax_available
def test_jax_array_summarizer_summarize_dtype(registry: SummarizerRegistry) -> None:
    result = JaxArraySummarizer().summarize(jnp.zeros(4, dtype=jnp.int32), registry)
    assert result.endswith(" | shape=(4,) | dtype=int32")


@jax_available
def test_jax_array_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    array = jnp.arange(3)
    assert JaxArraySummarizer(show_data=True).summarize(array, registry) == repr(array)
