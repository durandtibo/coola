from __future__ import annotations

from unittest.mock import Mock

from coola.summary import NDArraySummarizer, SummaryConfig
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

CONFIG = SummaryConfig()
SHOW_DATA = SummaryConfig(show_data=True)


#######################################
#     Tests for NDArraySummarizer     #
#######################################


@numpy_available
def test_ndarray_summarizer_summarize_1d_ndarray_default() -> None:
    """Test summarizing a 1D ndarray with default settings (metadata
    only)."""
    result = NDArraySummarizer().summarize(np.arange(11), CONFIG, compact=False)
    assert result == "<class 'numpy.ndarray'> | shape=(11,) | dtype=int64"


@numpy_available
def test_ndarray_summarizer_summarize_multidimensional_ndarray() -> None:
    """Test summarizing a multi-dimensional ndarray."""
    result = NDArraySummarizer().summarize(np.ones((2, 3, 4)), CONFIG, compact=False)
    assert result == "<class 'numpy.ndarray'> | shape=(2, 3, 4) | dtype=float64"


@numpy_available
def test_ndarray_summarizer_summarize_dtype_int32() -> None:
    result = NDArraySummarizer().summarize(
        np.array([1, 2, 3], dtype=np.int32), CONFIG, compact=False
    )
    assert result == "<class 'numpy.ndarray'> | shape=(3,) | dtype=int32"


@numpy_available
def test_ndarray_summarizer_summarize_dtype_float64() -> None:
    result = NDArraySummarizer().summarize(
        np.array([1.0, 2.0], dtype=np.float32), CONFIG, compact=False
    )
    assert result == "<class 'numpy.ndarray'> | shape=(2,) | dtype=float32"


@numpy_available
def test_ndarray_summarizer_summarize_dtype_bool() -> None:
    result = NDArraySummarizer().summarize(
        np.array([True, False], dtype=bool), CONFIG, compact=False
    )
    assert result == "<class 'numpy.ndarray'> | shape=(2,) | dtype=bool"


@numpy_available
def test_ndarray_summarizer_summarize_show_data_true() -> None:
    """Test summarizing with show_data=True returns full ndarray
    representation."""
    result = NDArraySummarizer().summarize(np.arange(5), SHOW_DATA, compact=False)
    assert result == "array([0, 1, 2, 3, 4])"


@numpy_available
def test_ndarray_summarizer_summarize_empty_ndarray() -> None:
    """Test summarizing an empty ndarray."""
    result = NDArraySummarizer().summarize(np.array([]), CONFIG, compact=False)
    assert result == "<class 'numpy.ndarray'> | shape=(0,) | dtype=float64"


@numpy_available
def test_ndarray_summarizer_summarize_scalar_ndarray() -> None:
    """Test summarizing a scalar ndarray (0-dimensional)."""
    result = NDArraySummarizer().summarize(np.array(42), CONFIG, compact=False)
    assert result == "<class 'numpy.ndarray'> | shape=() | dtype=int64"


@numpy_available
def test_ndarray_summarizer_summarize_large_ndarray() -> None:
    """Test summarizing a very large ndarray (metadata should still be
    compact)."""
    array = np.broadcast_to(np.float64(1), (1000, 1000, 10))  # no memory allocated
    result = NDArraySummarizer().summarize(array, CONFIG, compact=False)
    assert result == "<class 'numpy.ndarray'> | shape=(1000, 1000, 10) | dtype=float64"


@numpy_available
def test_ndarray_summarizer_summarize_masked_array() -> None:
    array = np.ma.masked_array([1, 2, 3], mask=[0, 1, 1])
    result = NDArraySummarizer().summarize(array, CONFIG, compact=False)
    assert result == "<class 'numpy.ma.MaskedArray'> | shape=(3,) | dtype=int64 | num_masked=2"


@numpy_available
def test_ndarray_summarizer_summarize_masked_array_without_mask() -> None:
    result = NDArraySummarizer().summarize(np.ma.masked_array([1, 2]), CONFIG, compact=False)
    assert result == "<class 'numpy.ma.MaskedArray'> | shape=(2,) | dtype=int64 | num_masked=0"


@numpy_available
def test_ndarray_summarizer_summarize_masked_arrays_with_a_different_mask_differ() -> None:
    summarizer = NDArraySummarizer()
    assert summarizer.summarize(
        np.ma.masked_array([1, 2], mask=[0, 1]), CONFIG, compact=False
    ) != summarizer.summarize(np.ma.masked_array([1, 2], mask=[0, 0]), CONFIG, compact=False)


@numpy_available
def test_ndarray_summarizer_summarize_ndarray_has_no_mask_field() -> None:
    assert "num_masked" not in NDArraySummarizer().summarize(np.ones(2), CONFIG, compact=False)


@numpy_available
def test_ndarray_summarizer_summarize_compact_is_ignored() -> None:
    summarizer = NDArraySummarizer()
    assert summarizer.summarize(np.arange(5), CONFIG, compact=True) == summarizer.summarize(
        np.arange(5), CONFIG, compact=False
    )
