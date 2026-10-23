from __future__ import annotations

import pytest

from tests.integration.helpers import run_python

PACKAGES = [
    "coola",
    "coola.equality",
    "coola.hashing",
    "coola.identifier",
    "coola.io",
    "coola.nested",
    "coola.random",
    "coola.reducer",
    "coola.summary",
    "coola.utils",
]
OPTIONAL_LIBRARIES = ("jax", "numpy", "pandas", "polars", "pyarrow", "torch", "xarray")

###################################################
#     Tests for the optional libraries imports    #
###################################################


@pytest.mark.parametrize("package", PACKAGES)
def test_importing_a_package_does_not_import_the_optional_libraries(package: str) -> None:
    """Test that importing a package does not load the optional
    libraries."""
    code = (
        f"import sys, {package}; "
        f"print(*[name for name in {OPTIONAL_LIBRARIES!r} if name in sys.modules])"
    )
    assert run_python(code) == []
