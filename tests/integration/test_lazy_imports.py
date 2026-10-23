from __future__ import annotations

import pytest

from tests.integration.helpers import run_python

OPTIONAL_LIBRARIES = ("jax", "numpy", "pandas", "polars", "pyarrow", "torch", "xarray")

###################################################
#     Tests for the optional libraries imports    #
###################################################


@pytest.mark.parametrize("package", ["coola", "coola.io", "coola.summary"])
def test_importing_a_package_does_not_import_the_optional_libraries(package: str) -> None:
    """Test that importing a package does not load the optional libraries."""
    code = (
        f"import sys, {package}; "
        f"print(*[name for name in {OPTIONAL_LIBRARIES!r} if name in sys.modules])"
    )
    assert run_python(code) == []


def test_importing_coola_io_does_not_import_coola_equality() -> None:
    """Test that importing ``coola.io`` does not load ``coola.equality``."""
    assert run_python("import sys, coola.io; print('coola.equality' in sys.modules)") == ["False"]
