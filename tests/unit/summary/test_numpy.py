from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola.summary import NDArraySummarizer, SummarizerRegistry
from coola.testing.fixtures import numpy_available
from coola.utils import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry()


######################################
#     Tests for NDArraySummarizer     #
######################################


@numpy_available
def test_ndarray_summarizer_init_default() -> None:
    """Test NDArraySummarizer initialization with default parameters."""
    summarizer = NDArraySummarizer()
    assert summarizer._show_data is False


@numpy_available
def test_ndarray_summarizer_init_show_data_true() -> None:
    """Test NDArraySummarizer initialization with show_data=True."""
    summarizer = NDArraySummarizer(show_data=True)
    assert summarizer._show_data is True


@numpy_available
def test_ndarray_summarizer_init_show_data_false() -> None:
    """Test NDArraySummarizer initialization with show_data=False."""
    summarizer = NDArraySummarizer(show_data=False)
    assert summarizer._show_data is False


@numpy_available
def test_ndarray_summarizer_repr_default() -> None:
    """Test __repr__ with default parameters."""
    assert repr(NDArraySummarizer()) == "NDArraySummarizer(show_data=False)"


@numpy_available
def test_ndarray_summarizer_repr_show_data_true() -> None:
    """Test __repr__ with show_data=True."""
    assert repr(NDArraySummarizer(show_data=True)) == "NDArraySummarizer(show_data=True)"


@numpy_available
def test_ndarray_summarizer_str_default() -> None:
    assert str(NDArraySummarizer()) == "NDArraySummarizer(show_data=False)"


@numpy_available
def test_ndarray_summarizer_str_show_data_true() -> None:
    assert str(NDArraySummarizer(show_data=True)) == "NDArraySummarizer(show_data=True)"


@numpy_available
def test_ndarray_summarizer_equal_same_instance() -> None:
    """Test equality between two summarizers with same configuration."""
    summarizer = NDArraySummarizer(show_data=False)
    assert summarizer.equal(summarizer)


@numpy_available
def test_ndarray_summarizer_equal_same_config() -> None:
    """Test equality between two summarizers with same configuration."""
    summarizer1 = NDArraySummarizer(show_data=False)
    summarizer2 = NDArraySummarizer(show_data=False)
    assert summarizer1.equal(summarizer2)


@numpy_available
def test_ndarray_summarizer_equal_different_show_data() -> None:
    """Test inequality between summarizers with different show_data."""
    summarizer1 = NDArraySummarizer(show_data=False)
    summarizer2 = NDArraySummarizer(show_data=True)
    assert not summarizer1.equal(summarizer2)


@numpy_available
def test_ndarray_summarizer_equal_different_type() -> None:
    """Test inequality when comparing with different type."""
    assert not NDArraySummarizer().equal(42)


@numpy_available
def test_ndarray_summarizer_equal_different_type_child() -> None:
    """Test inequality when comparing with different type."""

    class Child(NDArraySummarizer): ...

    assert not NDArraySummarizer().equal(Child())


@numpy_available
def test_ndarray_summarizer_summarize_1d_ndarray_default(registry: SummarizerRegistry) -> None:
    """Test summarizing a 1D ndarray with default settings (metadata
    only)."""
    result = NDArraySummarizer().summarize(registry, np.arange(11))
    assert result == "<class 'numpy.ndarray'> | shape=(11,) | dtype=int64"


@numpy_available
def test_ndarray_summarizer_summarize_multidimensional_ndarray(
    registry: SummarizerRegistry,
) -> None:
    """Test summarizing a multi-dimensional ndarray."""
    result = NDArraySummarizer().summarize(registry, np.ones((2, 3, 4)))
    assert result == "<class 'numpy.ndarray'> | shape=(2, 3, 4) | dtype=float64"


@numpy_available
def test_ndarray_summarizer_summarize_dtype_int32(registry: SummarizerRegistry) -> None:
    result = NDArraySummarizer().summarize(registry, np.array([1, 2, 3], dtype=np.int32))
    assert result == "<class 'numpy.ndarray'> | shape=(3,) | dtype=int32"


@numpy_available
def test_ndarray_summarizer_summarize_dtype_float64(registry: SummarizerRegistry) -> None:
    result = NDArraySummarizer().summarize(registry, np.array([1.0, 2.0], dtype=np.float32))
    assert result == "<class 'numpy.ndarray'> | shape=(2,) | dtype=float32"


@numpy_available
def test_ndarray_summarizer_summarize_dtype_bool(registry: SummarizerRegistry) -> None:
    result = NDArraySummarizer().summarize(registry, np.array([True, False], dtype=bool))
    assert result == "<class 'numpy.ndarray'> | shape=(2,) | dtype=bool"


@numpy_available
def test_ndarray_summarizer_summarize_show_data_true(registry: SummarizerRegistry) -> None:
    """Test summarizing with show_data=True returns full ndarray
    representation."""
    result = NDArraySummarizer(show_data=True).summarize(registry, np.arange(5))
    assert result == "array([0, 1, 2, 3, 4])"


@numpy_available
def test_ndarray_summarizer_summarize_empty_ndarray(registry: SummarizerRegistry) -> None:
    """Test summarizing an empty ndarray."""
    result = NDArraySummarizer().summarize(registry, np.array([]))
    assert result == "<class 'numpy.ndarray'> | shape=(0,) | dtype=float64"


@numpy_available
def test_ndarray_summarizer_summarize_scalar_ndarray(registry: SummarizerRegistry) -> None:
    """Test summarizing a scalar ndarray (0-dimensional)."""
    result = NDArraySummarizer().summarize(registry, np.array(42))
    assert result == "<class 'numpy.ndarray'> | shape=() | dtype=int64"


@numpy_available
def test_ndarray_summarizer_summarize_depth_ignored(registry: SummarizerRegistry) -> None:
    """Test that depth parameter is ignored."""
    summarizer = NDArraySummarizer()
    ndarray = np.arange(5)
    result1 = summarizer.summarize(registry, ndarray, depth=0)
    result2 = summarizer.summarize(registry, ndarray, depth=5)
    assert result1 == result2


@numpy_available
def test_ndarray_summarizer_summarize_max_depth_ignored(registry: SummarizerRegistry) -> None:
    """Test that max_depth parameter is ignored."""
    summarizer = NDArraySummarizer()
    ndarray = np.arange(5)
    result1 = summarizer.summarize(registry, ndarray, max_depth=1)
    result2 = summarizer.summarize(registry, ndarray, max_depth=10)
    assert result1 == result2


@numpy_available
def test_ndarray_summarizer_summarize_large_ndarray(registry: SummarizerRegistry) -> None:
    """Test summarizing a very large ndarray (metadata should still be
    compact)."""
    array = np.ones((1000, 1000, 10))
    result = NDArraySummarizer().summarize(registry, array)
    assert result == "<class 'numpy.ndarray'> | shape=(1000, 1000, 10) | dtype=float64"
