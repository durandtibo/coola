from __future__ import annotations

import pkgutil

import pytest

import coola
import coola.utils
from coola.testing.marks import numpy_available, polars_available, torch_available
from tests.integration.helpers import run_python

OPTIONAL_LIBRARIES = ("jax", "numpy", "pandas", "polars", "pyarrow", "pydantic", "torch", "xarray")
PACKAGES = ["coola", *(f"coola.{m.name}" for m in pkgutil.iter_modules(coola.__path__) if m.ispkg)]

UTILS_MODULES = [
    m.name for m in pkgutil.iter_modules(coola.utils.__path__) if m.name not in {"array", "tensor"}
]


def loaded_libraries(statement: str) -> list[str]:
    r"""Run a statement in a fresh interpreter and return the optional
    libraries that it imported."""
    code = (
        f"import sys; {statement}; "
        f"print(*[name for name in {OPTIONAL_LIBRARIES!r} if name in sys.modules])"
    )
    return run_python(code)


###################################################
#     Tests for the optional libraries imports    #
###################################################


@pytest.mark.parametrize("package", PACKAGES)
def test_importing_a_package_does_not_import_the_optional_libraries(package: str) -> None:
    """Test that importing a package does not load the optional
    libraries."""
    assert loaded_libraries(f"import {package}") == []


@pytest.mark.parametrize("module", ["package", "optional", "lazy"])
def test_importing_an_imports_module_does_not_import_the_optional_libraries(module: str) -> None:
    """Test that importing a module of ``coola.utils.imports`` does not
    load the optional libraries."""
    assert loaded_libraries(f"import coola.utils.imports.{module}") == []


@pytest.mark.parametrize("module", ["structure", "quantity", "text", "secret", "diff"])
def test_importing_a_formatting_module_does_not_import_the_optional_libraries(module: str) -> None:
    """Test that importing a module of ``coola.formatting`` does not
    load the optional libraries."""
    assert loaded_libraries(f"import coola.formatting.{module}") == []


@pytest.mark.parametrize("module", UTILS_MODULES)
def test_importing_a_utils_module_does_not_import_the_optional_libraries(module: str) -> None:
    """Test that importing a module of ``coola.utils`` other than
    ``array`` and ``tensor`` does not load the optional libraries."""
    assert loaded_libraries(f"import coola.utils.{module}") == []


def test_importing_a_lazy_name_imports_its_module() -> None:
    """Test that a name that needs an optional library is imported when
    it is used."""
    statement = "from coola.equality.tester import NumpyArrayEqualityTester"
    assert "numpy" in loaded_libraries(statement)


def test_a_lazy_name_is_in_the_dir_of_the_package() -> None:
    """Test that ``dir`` lists the names that are not imported yet."""
    code = "import coola.hashing; print('TorchTensorHasher' in dir(coola.hashing))"
    assert run_python(code) == ["True"]


def test_a_missing_name_raises_an_attribute_error() -> None:
    """Test that the lazy ``__getattr__`` rejects an unknown name."""
    code = "import coola.hashing; print(hasattr(coola.hashing, 'MissingHasher'))"
    assert run_python(code) == ["False"]


#################################################
#     Tests for the first use of the registries #
#################################################


@pytest.mark.parametrize(
    "statement",
    [
        "from coola.equality import objects_are_equal; objects_are_equal([1, {'a': 2}], [1, {'a': 2}])",
        "from coola.hashing import hash_object; hash_object([1, {'a': 2}])",
        "from coola.summary import summarize; summarize([1, {'a': 2}])",
        "from coola.jsonable import to_jsonable; to_jsonable([1, {'a': (2, {3})}])",
    ],
)
def test_using_the_registries_with_builtin_types_does_not_import_the_optional_libraries(
    statement: str,
) -> None:
    """Test that the default registries do not import the optional
    libraries for the builtin types."""
    assert loaded_libraries(statement) == []


@pytest.mark.parametrize(
    "statement",
    [
        "from coola.equality.tester import get_default_registry as f; repr(f())",
        "from coola.equality.tester import get_default_registry as f; str(f())",
        "from coola.hashing import get_default_registry as f; repr(f())",
        "from coola.summary import get_default_registry as f; repr(f())",
    ],
)
def test_displaying_a_default_registry_does_not_import_the_optional_libraries(
    statement: str,
) -> None:
    """Test that the repr of a default registry does not load the
    entries of the optional libraries."""
    assert loaded_libraries(statement) == []


@numpy_available
@pytest.mark.parametrize(
    "statement",
    [
        "from coola.equality import objects_are_equal; objects_are_equal(np.ones(2), np.ones(2))",
        "from coola.hashing import hash_object; hash_object(np.ones(2))",
        "from coola.summary import summarize; summarize(np.ones(2))",
    ],
)
def test_using_the_registries_with_a_numpy_array_imports_only_numpy(statement: str) -> None:
    """Test that the registries load only the entries of the library of
    the object."""
    assert loaded_libraries(f"import numpy as np; {statement}") == ["numpy"]


@torch_available
def test_using_the_registries_with_a_tensor_does_not_load_the_entries_of_pandas() -> None:
    """Test that the entries of the libraries that are not imported are
    not loaded."""
    statement = (
        "import torch; from coola.summary import summarize; summarize(torch.ones(2)); "
        "from coola.equality import objects_are_equal; objects_are_equal(torch.ones(2), "
        "torch.ones(2))"
    )
    assert "pandas" not in loaded_libraries(statement)


@numpy_available
def test_manual_seed_seeds_numpy_even_if_it_is_imported_after() -> None:
    """Test that the random state of a library that is imported after
    ``manual_seed`` is seeded."""
    code = (
        "from coola.random import manual_seed; manual_seed(42); import numpy as np; "
        "print(np.random.rand())"
    )
    assert run_python(code) == run_python(code)


def test_importing_coola_does_not_import_importlib_metadata() -> None:
    """Test that the version is read only when it is used."""
    code = "import sys, coola; print('importlib.metadata' in sys.modules)"
    assert run_python(code) == ["False"]


def test_using_the_version_imports_importlib_metadata() -> None:
    """Test that ``__version__`` is read from the metadata."""
    code = "import sys, coola; coola.__version__; print('importlib.metadata' in sys.modules)"
    assert run_python(code) == ["True"]


@polars_available
def test_importing_flatten_frame_imports_polars() -> None:
    """Test that ``flatten_frame`` loads polars the first time it is
    used, not when ``coola.nested`` is imported."""
    assert loaded_libraries("from coola.nested import flatten_frame") == ["polars"]


def test_flatten_frame_is_in_the_dir_of_nested() -> None:
    """Test that ``dir`` lists ``flatten_frame`` before it is
    imported."""
    code = "import coola.nested; print('flatten_frame' in dir(coola.nested))"
    assert run_python(code) == ["True"]


def test_rounding_a_float_does_not_import_the_optional_libraries() -> None:
    """Test that rounding a native float does not load the optional
    libraries."""
    assert (
        loaded_libraries("from coola.rounding import round_values; round_values(1.5, decimals=1)")
        == []
    )
