from __future__ import annotations

import importlib
import logging

from coola import objects_are_allclose, objects_are_equal
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
    import polars
if is_torch_available():
    import torch
if is_xarray_available():
    import xarray as xr

logger = logging.getLogger(__name__)


def check_imports() -> None:
    logger.info("Checking imports...")
    objects_to_import = [
        "coola.equality.comparators.BaseEqualityComparator",
        "coola.equality.testers.BaseEqualityTester",
        "coola.formatters.BaseFormatter",
        "coola.objects_are_allclose",
        "coola.objects_are_equal",
        "coola.reducers.BaseReducer",
        "coola.summarizers.BaseSummarizer",
        "coola.summary",
    ]
    for a in objects_to_import:
        module_path, name = a.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
        obj = getattr(module, name)
        assert obj is not None


def check_native_comparators() -> None:
    logger.info("Checking native comparators...")
    assert objects_are_allclose("cat", "cat")
    assert objects_are_allclose(4.2, 4.2)
    assert objects_are_allclose([4.2, 2], [4.2, 2])
    assert objects_are_allclose({"list": [4.2, 2], "float": 2.1}, {"list": [4.2, 2], "float": 2.1})

    assert objects_are_equal("cat", "cat")
    assert objects_are_equal(4.2, 4.2)
    assert objects_are_equal([4.2, 2], [4.2, 2])
    assert objects_are_equal({"list": [4.2, 2], "float": 2.1}, {"list": [4.2, 2], "float": 2.1})


@jax_available
def check_jax_comparators() -> None:
    logger.info("Checking jax comparators...")
    assert is_jax_available()
    assert objects_are_allclose(jnp.ones((2, 3)), jnp.ones((2, 3)))
    assert objects_are_equal(jnp.ones((2, 3)), jnp.ones((2, 3)))


@numpy_available
def check_numpy_comparators() -> None:
    logger.info("Checking numpy comparators...")
    assert is_numpy_available()
    assert objects_are_allclose(np.ones((2, 3)), np.ones((2, 3)))
    assert objects_are_equal(np.ones((2, 3)), np.ones((2, 3)))


@pandas_available
def check_pandas_comparators() -> None:
    logger.info("Checking pandas comparators...")
    assert is_pandas_available()
    assert objects_are_allclose(
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pd.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pd.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )
    assert objects_are_equal(
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pd.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pd.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@polars_available
def check_polars_comparators() -> None:
    logger.info("Checking polars comparators...")
    assert is_polars_available()
    assert objects_are_allclose(
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )
    assert objects_are_equal(
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@torch_available
def check_torch_comparators() -> None:
    logger.info("Checking torch comparators...")
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
def check_xarray_comparators() -> None:
    logger.info("Checking xarray comparators...")
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


def main() -> None:
    check_imports()
    check_native_comparators()
    check_jax_comparators()
    check_numpy_comparators()
    check_pandas_comparators()
    check_polars_comparators()
    check_torch_comparators()
    check_xarray_comparators()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
