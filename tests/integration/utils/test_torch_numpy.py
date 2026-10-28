from __future__ import annotations

from coola.testing.marks import torch_available
from tests.integration.helpers import run_python

################################################
#     Tests for the imports of torch+numpy     #
################################################


def test_is_torch_numpy_available_does_not_import_torch_if_numpy_is_missing() -> None:
    code = (
        "import sys; "
        "from unittest.mock import patch; "
        "import coola.utils.imports.optional as m; "
        "p = patch.object(m, 'is_torch_available', lambda: True); "
        "q = patch.object(m, 'is_numpy_available', lambda: False); "
        "p.start(); q.start(); "
        "print(m.is_torch_numpy_available(), 'torch' in sys.modules)"
    )
    assert run_python(code) == ["False", "False"]


def test_importing_coola_utils_imports_does_not_import_torch_nor_numpy() -> None:
    code = "import sys, coola.utils.imports; print('torch' in sys.modules, 'numpy' in sys.modules)"
    assert run_python(code) == ["False", "False"]


def test_importing_the_availability_checks_does_not_import_torch() -> None:
    code = (
        "import sys; "
        "from coola.utils.imports import is_torch_numpy_available, check_torch_numpy; "
        "print('torch' in sys.modules)"
    )
    assert run_python(code) == ["False"]


@torch_available
def test_is_torch_numpy_available_imports_torch_when_called() -> None:
    code = (
        "import sys; "
        "from coola.utils.imports import is_torch_numpy_available; "
        "is_torch_numpy_available(); "
        "print('torch' in sys.modules)"
    )
    assert run_python(code)[0] == "True"
