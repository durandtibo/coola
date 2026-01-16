from __future__ import annotations

import importlib
import logging
import random
import sys

from coola.equality import objects_are_allclose, objects_are_equal
from coola.iterator import bfs_iterate, dfs_iterate
from coola.random import manual_seed
from coola.recursive import recursive_apply
from coola.reducer import NativeReducer
from coola.registry import Registry
from coola.summary import summarize
from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
    is_xarray_available,
    jax_available,
    numpy_available,
    pandas_available,
    polars_available,
    torch_available,
    xarray_available,
)

if is_jax_available():
    import jax.numpy as jnp
if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_torch_available():
    import torch
if is_xarray_available():
    import xarray as xr

logger = logging.getLogger(__name__)


def check_imports() -> None:
    r"""Check that all main package objects can be imported
    correctly."""
    logger.info("Checking imports...")
    objects_to_import = [
        "coola.equality.config.EqualityConfig",
        "coola.equality.handler.BaseEqualityHandler",
        "coola.equality.objects_are_allclose",
        "coola.equality.objects_are_equal",
        "coola.equality.tester.BaseEqualityTester",
        "coola.iterator.bfs.BaseChildFinder",
        "coola.iterator.bfs.ChildFinderRegistry",
        "coola.iterator.bfs_iterate",
        "coola.iterator.dfs.BaseIterator",
        "coola.iterator.dfs.IteratorRegistry",
        "coola.iterator.dfs_iterate",
        "coola.nested.to_flat_dict",
        "coola.random.BaseRandomManager",
        "coola.random.RandomManagerRegistry",
        "coola.random.random_seed",
        "coola.recursive.BaseTransformer",
        "coola.recursive.TransformerRegistry",
        "coola.recursive.recursive_apply",
        "coola.reducer.BaseReducer",
        "coola.registry.Registry",
        "coola.registry.TypeRegistry",
        "coola.summary.BaseSummarizer",
        "coola.summary.SummarizerRegistry",
        "coola.summary.summarize",
    ]
    for a in objects_to_import:
        module_path, name = a.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
        obj = getattr(module, name)
        assert obj is not None, f"Failed to import {a}"


def check_equality_native() -> None:
    r"""Check equality operations for native Python types."""
    logger.info("Checking native equality testers...")
    assert objects_are_allclose("cat", "cat")
    assert objects_are_allclose(4.2, 4.2)
    assert objects_are_allclose([4.2, 2], [4.2, 2])
    assert objects_are_allclose({"list": [4.2, 2], "float": 2.1}, {"list": [4.2, 2], "float": 2.1})

    assert objects_are_equal("cat", "cat")
    assert objects_are_equal(4.2, 4.2)
    assert objects_are_equal([4.2, 2], [4.2, 2])
    assert objects_are_equal({"list": [4.2, 2], "float": 2.1}, {"list": [4.2, 2], "float": 2.1})


@jax_available
def check_equality_jax() -> None:
    r"""Check equality operations for JAX arrays."""
    logger.info("Checking jax equality testers...")
    assert is_jax_available()
    assert objects_are_allclose(jnp.ones((2, 3)), jnp.ones((2, 3)))
    assert objects_are_equal(jnp.ones((2, 3)), jnp.ones((2, 3)))


@numpy_available
def check_equality_numpy() -> None:
    r"""Check equality operations for NumPy arrays."""
    logger.info("Checking numpy equality testers...")
    assert is_numpy_available()
    assert objects_are_allclose(np.ones((2, 3)), np.ones((2, 3)))
    assert objects_are_equal(np.ones((2, 3)), np.ones((2, 3)))


@pandas_available
def check_equality_pandas() -> None:
    r"""Check equality operations for pandas DataFrames."""
    logger.info("Checking pandas equality testers...")
    assert is_pandas_available()
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": pd.to_datetime(
                ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
            ),
        }
    )
    assert objects_are_allclose(df, df.copy())
    assert objects_are_equal(df, df.copy())


@polars_available
def check_equality_polars() -> None:
    r"""Check equality operations for Polars DataFrames."""
    logger.info("Checking polars equality testers...")
    assert is_polars_available()
    df = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": pl.Series(
                ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
            ).str.to_datetime(),
        }
    )
    assert objects_are_allclose(df, df.clone())
    assert objects_are_equal(df, df.clone())


@torch_available
def check_equality_torch() -> None:
    r"""Check equality operations for PyTorch tensors and structures."""
    logger.info("Checking torch equality testers...")
    assert is_torch_available()
    # Tensor
    assert objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3))
    assert objects_are_equal(torch.ones(2, 3), torch.ones(2, 3))

    # PackedSequence
    assert objects_are_allclose(
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )
    assert objects_are_equal(
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
    )


@xarray_available
def check_equality_xarray() -> None:
    r"""Check equality operations for xarray DataArrays, Datasets, and
    Variables."""
    logger.info("Checking xarray equality testers...")
    assert is_xarray_available()
    # DataArray
    assert objects_are_allclose(
        xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"])
    )
    assert objects_are_equal(
        xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"])
    )

    # Dataset
    assert objects_are_allclose(
        xr.Dataset(
            {
                "x": xr.DataArray(np.arange(6), dims=["z"]),
                "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
            },
            coords={"z": np.arange(6) + 1, "t": [1, 2, 3]},
            attrs={"global": "this is a global attribute"},
        ),
        xr.Dataset(
            {
                "x": xr.DataArray(np.arange(6), dims=["z"]),
                "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
            },
            coords={"z": np.arange(6) + 1, "t": [1, 2, 3]},
            attrs={"global": "this is a global attribute"},
        ),
    )
    assert objects_are_equal(
        xr.Dataset(
            {
                "x": xr.DataArray(np.arange(6), dims=["z"]),
                "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
            },
            coords={"z": np.arange(6) + 1, "t": [1, 2, 3]},
            attrs={"global": "this is a global attribute"},
        ),
        xr.Dataset(
            {
                "x": xr.DataArray(np.arange(6), dims=["z"]),
                "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
            },
            coords={"z": np.arange(6) + 1, "t": [1, 2, 3]},
            attrs={"global": "this is a global attribute"},
        ),
    )

    # Variable
    assert objects_are_allclose(
        xr.Variable(dims=["z"], data=np.arange(6)), xr.Variable(dims=["z"], data=np.arange(6))
    )
    assert objects_are_equal(
        xr.Variable(dims=["z"], data=np.arange(6)), xr.Variable(dims=["z"], data=np.arange(6))
    )


def check_iterator_bfs() -> None:
    r"""Check breadth-first search iteration over nested structures."""
    logger.info("Checking iterator BFS...")
    assert objects_are_equal(
        list(bfs_iterate({"a": {"b": 2, "c": {"d": 1, "e": 4}}, "d": 3})), [3, 2, 1, 4]
    )


def check_iterator_dfs() -> None:
    r"""Check depth-first search iteration over nested structures."""
    logger.info("Checking iterator DFS...")
    assert objects_are_equal(
        list(
            dfs_iterate(
                {"a": {"b": [1, 2], "c": 3}, "d": 4},
            )
        ),
        [1, 2, 3, 4],
    )


def check_random() -> None:
    r"""Check random seed management functionality."""
    logger.info("Checking random managers...")
    manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    manual_seed(42)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2


def check_recursive() -> None:
    r"""Check recursive application of transformations."""
    logger.info("Checking recursive...")
    assert objects_are_equal(recursive_apply([1, 2, 3], lambda x: x * 2), [2, 4, 6])


def check_reducer() -> None:
    r"""Check reducer operations for native sequences."""
    logger.info("Checking reducer...")
    reducer = NativeReducer()
    assert reducer.max([-2, -1, 0, 1, 2]) == 2
    assert reducer.median([-2, -1, 0, 1, 2]) == 0
    assert reducer.sort([2, 1, -2, 3, 0]) == [-2, 0, 1, 2, 3]


def check_registry() -> None:
    r"""Check registry functionality."""
    logger.info("Checking registry...")
    registry = Registry[str, int]()
    registry.register("a", 1)
    registry.register("b", 2)
    registry.register("c", 3)
    assert registry.equal(Registry[str, int]({"a": 1, "b": 2, "c": 3}))


def check_summary() -> None:
    r"""Check object summarization functionality."""
    logger.info("Checking summary...")
    expected = "<class 'dict'> (length=2)\n  (a): 1\n  (b): 2"
    actual = summarize({"a": 1, "b": 2})
    assert actual == expected


def main() -> None:
    r"""Run all package checks to validate installation and
    functionality."""
    try:
        check_imports()
        check_equality_native()
        check_equality_jax()
        check_equality_numpy()
        check_equality_pandas()
        check_equality_polars()
        check_equality_torch()
        check_equality_xarray()

        check_iterator_bfs()
        check_iterator_dfs()

        check_random()
        check_recursive()
        check_reducer()
        check_registry()
        check_summary()

        logger.info("✅ All package checks passed successfully!")
    except Exception:
        logger.exception("❌ Package check failed")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
