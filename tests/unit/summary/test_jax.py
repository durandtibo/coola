from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    JaxArraySummarizer,
    SummaryConfig,
)
from coola.testing.marks import jax_available
from coola.utils.imports import MissingPackageError, is_jax_available

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()


SUMMARIZERS = [
    JaxArraySummarizer,
]

CONFIG = SummaryConfig()
SHOW_DATA = SummaryConfig(show_data=True)


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.optional.is_jax_available", lambda: False),
        pytest.raises(MissingPackageError, match="'jax' package is required"),
    ):
        summarizer_cls()


@jax_available
def test_jax_array_summarizer_summarize() -> None:
    result = JaxArraySummarizer().summarize(jnp.ones((2, 3)), CONFIG, compact=False)
    assert result.endswith(" | shape=(2, 3) | dtype=float32")


@jax_available
def test_jax_array_summarizer_summarize_dtype() -> None:
    result = JaxArraySummarizer().summarize(jnp.zeros(4, dtype=jnp.int32), CONFIG, compact=False)
    assert result.endswith(" | shape=(4,) | dtype=int32")


@jax_available
def test_jax_array_summarizer_summarize_show_data() -> None:
    array = jnp.arange(3)
    assert JaxArraySummarizer().summarize(array, SHOW_DATA, compact=False) == repr(array)
