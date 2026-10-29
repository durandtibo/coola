r"""Check that the installed ``coola`` package works.

The script is run by the CI against a package that is *installed* (built
from the source distribution or the wheel, or downloaded from PyPI), not
against the source tree. It is a standalone script, not a pytest module:
it must run in an environment where only ``coola`` (and the requested
extras) are installed, so it does not import ``pytest`` or
``coola.testing``, and it does not import an optional library at module
level.

Usage::

    python tests/package/checks.py                  # no extra
    python tests/package/checks.py --extra numpy    # numpy is required
    python tests/package/checks.py --extra all      # every extra is required

A check on an optional library is skipped (and reported as skipped) when
the library is not installed, unless the extra that provides it was
requested with ``--extra``: then a missing library is a failure.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import pkgutil
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import coola
from coola.equality import objects_are_allclose, objects_are_equal
from coola.iterator import bfs_iterate, dfs_iterate
from coola.nested import to_flat_dict
from coola.random import manual_seed
from coola.recursive import recursive_apply
from coola.reducer import NativeReducer
from coola.registry import Registry
from coola.summary import summarize

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import FunctionType

logger = logging.getLogger(__name__)

PACKAGE = "coola"

# Extra name -> module that the extra installs. The extras that are only
# installed on some platforms are in ``PLATFORM_DEPENDENT_EXTRAS``.
EXTRA_MODULES = {
    "jax": "jax",
    "numpy": "numpy",
    "packaging": "packaging",
    "pandas": "pandas",
    "polars": "polars",
    "pyarrow": "pyarrow",
    "pydantic": "pydantic",
    "torch": "torch",
    "xarray": "xarray",
    "colorlog": "colorlog",
    "rich": "rich",
}

# These extras have environment markers (OS, architecture, Python version) in
# ``pyproject.toml``: on an unsupported platform the extra installs nothing,
# so a missing library is not a failure.
PLATFORM_DEPENDENT_EXTRAS = frozenset({"jax", "torch"})

# Modules that must not be imported by ``import coola.<module>``.
LAZY_MODULES = ("pytest", *sorted(set(EXTRA_MODULES.values()) - {"packaging"}))

# ``coola.testing`` imports ``pytest``, which is only installed by the ``testing`` extra.
IMPORT_EXCLUDED = ("coola.testing",)

REQUIRED_EXTRAS: set[str] = set()


class SkipCheckError(Exception):
    r"""Raised by a check that cannot run in this environment."""


def is_installed(module: str) -> bool:
    r"""Indicate if a module can be imported, without importing it."""
    return importlib.util.find_spec(module) is not None


def require(extra: str) -> None:
    r"""Make sure a check can run with the library of an extra.

    Args:
        extra: The name of the extra.

    Raises:
        SkipCheckError: If the library is not installed and the extra
            was not requested.
        ModuleNotFoundError: If the library is not installed and the
            extra was requested.
    """
    module = EXTRA_MODULES[extra]
    if is_installed(module):
        return
    if extra in REQUIRED_EXTRAS and extra not in PLATFORM_DEPENDENT_EXTRAS:
        msg = f"the extra {extra!r} was requested but {module!r} is not installed"
        raise ModuleNotFoundError(msg)
    msg = f"{module!r} is not installed"
    raise SkipCheckError(msg)


#################################
#     Checks of the package     #
#################################


def is_editable_install() -> bool:
    r"""Indicate if ``coola`` is installed in editable mode (PEP
    610)."""
    text = importlib.metadata.distribution(PACKAGE).read_text("direct_url.json")
    return bool(text) and bool(json.loads(text or "{}").get("dir_info", {}).get("editable"))


def check_install_location() -> None:
    r"""Check that ``coola`` is imported from the installed package, not
    from the source tree.

    An editable install is imported from the source tree by design, so
    the check is skipped.
    """
    if is_editable_install():
        msg = "editable install"
        raise SkipCheckError(msg)

    location = Path(coola.__file__).resolve()
    source_tree = Path(__file__).resolve().parents[2] / "src"
    assert source_tree not in location.parents, (
        f"coola is imported from the source tree: {location}"
    )


def check_metadata() -> None:
    r"""Check the metadata of the installed distribution."""
    metadata = importlib.metadata.metadata(PACKAGE)
    assert metadata["Name"] == PACKAGE
    assert metadata["Requires-Python"], "Requires-Python is missing"
    assert metadata["License-Expression"] or metadata["License"], "the license is missing"
    installed = importlib.metadata.version(PACKAGE)
    assert installed != "0.0.0", "the version is the fallback of an uninstalled package"
    assert coola.__version__ == installed, f"{coola.__version__!r} != {installed!r}"


def check_distribution_files() -> None:
    r"""Check that the typing marker and the license are shipped."""
    assert (Path(coola.__file__).parent / "py.typed").is_file(), "py.typed is missing"
    files = {str(f) for f in importlib.metadata.files(PACKAGE) or []}
    assert any(f.endswith("LICENSE") for f in files), "LICENSE is missing"


def check_extras_installed() -> None:
    r"""Check that the distribution of each requested extra is
    installed."""
    missing = []
    for extra in sorted(REQUIRED_EXTRAS - PLATFORM_DEPENDENT_EXTRAS):
        try:
            importlib.metadata.version(extra)
        except importlib.metadata.PackageNotFoundError:
            missing.append(extra)
    assert not missing, f"the packages of the extras are not installed: {missing}"


def check_import_all_modules() -> None:
    r"""Check that every module of the package can be imported.

    This catches a module that is missing from the sdist or the wheel. A
    module of a backend (for example ``coola.display.colorlog``) imports
    its library, so it may fail with ``ModuleNotFoundError`` if the
    library is not installed and its extra was not requested.
    """
    package = importlib.import_module(PACKAGE)
    required = {EXTRA_MODULES[e] for e in REQUIRED_EXTRAS}
    failures = []
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{PACKAGE}."):
        if info.name.startswith(IMPORT_EXCLUDED):
            continue
        try:
            importlib.import_module(info.name)
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            if missing not in EXTRA_MODULES.values() or missing in required:
                failures.append(f"{info.name}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{info.name}: {type(exc).__name__}: {exc}")
    assert not failures, "failed to import:\n" + "\n".join(failures)


def check_lazy_imports() -> None:
    r"""Check that importing the subpackages does not import the
    optional libraries (nor ``pytest``)."""
    package = importlib.import_module(PACKAGE)
    for info in pkgutil.iter_modules(package.__path__, prefix=f"{PACKAGE}."):
        if not info.name.startswith(IMPORT_EXCLUDED):
            importlib.import_module(info.name)
    imported = [name for name in LAZY_MODULES if name in sys.modules]
    assert not imported, f"imported by the package at import time: {imported}"


def check_imports() -> None:
    r"""Check that all main package objects can be imported
    correctly."""
    objects_to_import = [
        "coola.equality.config.EqualityConfig",
        "coola.equality.Engine",
        "coola.equality.objects_are_allclose",
        "coola.equality.objects_are_equal",
        "coola.equality.tester.BaseEqualityTester",
        "coola.iterator.BaseChildFinder",
        "coola.iterator.ChildFinderRegistry",
        "coola.iterator.bfs_iterate",
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
        assert getattr(module, name, None) is not None, f"Failed to import {a}"


################################
#     Checks of the features   #
################################


def check_equality_native() -> None:
    r"""Check equality operations for native Python types."""
    for func in (objects_are_allclose, objects_are_equal):
        assert func("cat", "cat")
        assert func(4.2, 4.2)
        assert func([4.2, 2], [4.2, 2])
        assert func({"list": [4.2, 2], "float": 2.1}, {"list": [4.2, 2], "float": 2.1})
        assert not func([4.2, 2], [4.2, 3])


def check_equality_jax() -> None:
    r"""Check equality operations for JAX arrays."""
    require("jax")
    jnp = importlib.import_module("jax.numpy")

    assert objects_are_allclose(jnp.ones((2, 3)), jnp.ones((2, 3)))
    assert objects_are_equal(jnp.ones((2, 3)), jnp.ones((2, 3)))
    assert not objects_are_equal(jnp.ones((2, 3)), jnp.zeros((2, 3)))


def check_equality_numpy() -> None:
    r"""Check equality operations for NumPy arrays."""
    require("numpy")
    np = importlib.import_module("numpy")

    assert objects_are_allclose(np.ones((2, 3)), np.ones((2, 3)))
    assert objects_are_equal(np.ones((2, 3)), np.ones((2, 3)))
    assert not objects_are_equal(np.ones((2, 3)), np.zeros((2, 3)))


def check_equality_pandas() -> None:
    r"""Check equality operations for pandas DataFrames."""
    require("pandas")
    pd = importlib.import_module("pandas")

    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
        }
    )
    assert objects_are_allclose(df, df.copy())
    assert objects_are_equal(df, df.copy())
    assert not objects_are_equal(df, df.assign(col1=[0, 2, 3, 4, 5]))


def check_equality_polars() -> None:
    r"""Check equality operations for Polars DataFrames."""
    require("polars")
    pl = importlib.import_module("polars")

    df = pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
        }
    )
    assert objects_are_allclose(df, df.clone())
    assert objects_are_equal(df, df.clone())
    assert not objects_are_equal(df, df.with_columns(pl.lit(0).alias("col1")))


def check_equality_pyarrow() -> None:
    r"""Check equality operations for PyArrow arrays and tables."""
    require("pyarrow")
    pa = importlib.import_module("pyarrow")

    table = pa.table({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    assert objects_are_equal(table, deepcopy(table))
    assert objects_are_equal(pa.array([1, 2, 3]), pa.array([1, 2, 3]))
    assert not objects_are_equal(pa.array([1, 2, 3]), pa.array([1, 2, 4]))


def check_equality_torch() -> None:
    r"""Check equality operations for PyTorch tensors and structures."""
    require("torch")
    torch = importlib.import_module("torch")

    assert objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3))
    assert objects_are_equal(torch.ones(2, 3), torch.ones(2, 3))
    assert not objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3))

    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
        input=torch.arange(10, dtype=torch.float).view(2, 5),
        lengths=torch.tensor([5, 3], dtype=torch.long),
        batch_first=True,
    )
    assert objects_are_allclose(packed_sequence, deepcopy(packed_sequence))
    assert objects_are_equal(packed_sequence, deepcopy(packed_sequence))


def check_equality_xarray() -> None:
    r"""Check equality operations for xarray DataArrays, Datasets, and
    Variables."""
    require("xarray")
    require("numpy")
    np = importlib.import_module("numpy")
    xr = importlib.import_module("xarray")

    data = xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"])
    assert objects_are_allclose(data, deepcopy(data))
    assert objects_are_equal(data, deepcopy(data))

    dataset = xr.Dataset(
        {
            "x": xr.DataArray(np.arange(6), dims=["z"]),
            "y": xr.DataArray(np.ones((6, 3)), dims=["z", "t"]),
        },
        coords={"z": np.arange(6) + 1, "t": [1, 2, 3]},
        attrs={"global": "this is a global attribute"},
    )
    assert objects_are_allclose(dataset, deepcopy(dataset))
    assert objects_are_equal(dataset, deepcopy(dataset))

    variable = xr.Variable(dims=["z"], data=np.arange(6))
    assert objects_are_allclose(variable, deepcopy(variable))
    assert objects_are_equal(variable, deepcopy(variable))


def check_iterator_bfs() -> None:
    r"""Check breadth-first search iteration over nested structures."""
    assert objects_are_equal(
        list(bfs_iterate({"a": {"b": 2, "c": {"d": 1, "e": 4}}, "d": 3})), [3, 2, 1, 4]
    )


def check_iterator_dfs() -> None:
    r"""Check depth-first search iteration over nested structures."""
    assert objects_are_equal(list(dfs_iterate({"a": {"b": [1, 2], "c": 3}, "d": 4})), [1, 2, 3, 4])


def check_nested() -> None:
    r"""Check flattening of nested structures."""
    assert to_flat_dict({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


def check_random() -> None:
    r"""Check random seed management functionality."""
    manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    manual_seed(42)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2


def check_recursive() -> None:
    r"""Check recursive application of transformations."""
    assert objects_are_equal(recursive_apply([1, 2, 3], lambda x: x * 2), [2, 4, 6])


def check_reducer() -> None:
    r"""Check reducer operations for native sequences."""
    reducer = NativeReducer()
    assert reducer.max([-2, -1, 0, 1, 2]) == 2
    assert reducer.median([-2, -1, 0, 1, 2]) == 0
    assert reducer.sort([2, 1, -2, 3, 0]) == [-2, 0, 1, 2, 3]


def check_registry() -> None:
    r"""Check registry functionality."""
    registry = Registry[str, int]()
    registry.register("a", 1)
    registry.register("b", 2)
    registry.register("c", 3)
    assert registry.equal(Registry[str, int]({"a": 1, "b": 2, "c": 3}))


def check_summary() -> None:
    r"""Check object summarization functionality."""
    assert summarize({"a": 1, "b": 2}) == "<class 'dict'> (length=2)\n  (a): 1\n  (b): 2"


# ``check_lazy_imports`` is first: it must run before the other checks import
# the optional libraries.
CHECKS: tuple[FunctionType, ...] = (
    check_install_location,
    check_metadata,
    check_distribution_files,
    check_extras_installed,
    check_lazy_imports,
    check_import_all_modules,
    check_imports,
    check_equality_native,
    check_equality_jax,
    check_equality_numpy,
    check_equality_pandas,
    check_equality_polars,
    check_equality_pyarrow,
    check_equality_torch,
    check_equality_xarray,
    check_iterator_bfs,
    check_iterator_dfs,
    check_nested,
    check_random,
    check_recursive,
    check_reducer,
    check_registry,
    check_summary,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    r"""Parse the command line arguments.

    Args:
        argv: The arguments. ``None`` means ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="extra installed with the package: its library must be importable. "
        "Can be repeated or comma-separated; 'all' means every extra; '' means none.",
    )
    return parser.parse_args(argv)


def get_required_extras(values: Sequence[str]) -> set[str]:
    r"""Get the extras whose library is required.

    Args:
        values: The values of ``--extra``.

    Returns:
        The names of the extras.
    """
    extras = {e.strip() for value in values for e in value.split(",") if e.strip()}
    if "all" in extras:
        extras = (extras - {"all"}) | set(EXTRA_MODULES)
    # An extra without an optional library (for example ``testing``) has nothing to check.
    extras &= set(EXTRA_MODULES)
    return extras


def main(argv: Sequence[str] | None = None) -> int:
    r"""Run all the package checks.

    Args:
        argv: The command line arguments.

    Returns:
        The exit code: ``0`` if no check failed, ``1`` otherwise.
    """
    REQUIRED_EXTRAS.update(get_required_extras(parse_args(argv).extra))
    logger.info("Required extras: %s", sorted(REQUIRED_EXTRAS) or "none")
    failed, skipped = [], []
    for check in CHECKS:
        name = check.__name__
        try:
            check()
        except SkipCheckError as exc:
            skipped.append(name)
            logger.info("⏭️  %s skipped: %s", name, exc)
        except Exception:
            failed.append(name)
            logger.exception("❌ %s failed", name)
        else:
            logger.info("✅ %s passed", name)
    passed = len(CHECKS) - len(failed) - len(skipped)
    logger.info("%d passed, %d skipped, %d failed", passed, len(skipped), len(failed))
    if failed:
        logger.error("❌ Package check failed: %s", ", ".join(failed))
        return 1
    logger.info("✅ All package checks passed successfully!")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
