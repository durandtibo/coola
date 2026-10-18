from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

import coola
from coola.testing.fixtures import torch_available
from coola.utils.imports import (
    MissingPackageError,
    check_torch_numpy,
    is_torch_numpy_available,
    torch_numpy_available,
)

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

MODULE = "coola.utils.imports.torch_numpy"


@pytest.fixture(autouse=True)
def _cache_clear() -> Generator[None]:
    is_torch_numpy_available.cache_clear()
    yield
    is_torch_numpy_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


def run_python(code: str) -> list[str]:
    r"""Run ``code`` in a fresh interpreter (the modules imported by
    this process would hide the imports of ``code``) and return the
    words it prints."""
    src = str(Path(coola.__file__).parents[1])
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": src},
    )
    return result.stdout.split()


#######################
#     torch+numpy     #
#######################


def test_check_torch_numpy() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: True):
        check_torch_numpy()


def test_check_torch_numpy_missing() -> None:
    with (
        patch(f"{MODULE}.is_torch_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'torch' and 'numpy' packages are required"),
    ):
        check_torch_numpy()


def test_is_torch_numpy_available() -> None:
    assert isinstance(is_torch_numpy_available(), bool)


def test_is_torch_numpy_available_missing_torch() -> None:
    with patch(f"{MODULE}.is_torch_available", lambda: False):
        assert not is_torch_numpy_available()


def test_is_torch_numpy_available_missing_numpy() -> None:
    with (
        patch(f"{MODULE}.is_torch_available", lambda: True),
        patch(f"{MODULE}.is_numpy_available", lambda: False),
    ):
        assert not is_torch_numpy_available()


@torch_available
def test_is_torch_numpy_available_not_compatible() -> None:
    with (
        patch(f"{MODULE}.is_torch_available", lambda: True),
        patch(f"{MODULE}.is_numpy_available", lambda: True),
        patch("torch.tensor", Mock(side_effect=RuntimeError)),
    ):
        assert not is_torch_numpy_available()


@torch_available
def test_is_torch_numpy_available_compatible() -> None:
    with (
        patch(f"{MODULE}.is_torch_available", lambda: True),
        patch(f"{MODULE}.is_numpy_available", lambda: True),
        patch("torch.tensor", Mock(return_value=Mock(numpy=Mock(return_value=[1.0])))),
    ):
        assert is_torch_numpy_available()


def test_torch_numpy_available_with_package() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: True):
        fn = torch_numpy_available(my_function)
        assert fn(2) == 44


def test_torch_numpy_available_without_package() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: False):
        fn = torch_numpy_available(my_function)
        assert fn(2) is None


def test_torch_numpy_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: True):

        @torch_numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_torch_numpy_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: False):

        @torch_numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_is_torch_numpy_available_does_not_import_torch_if_numpy_is_missing() -> None:
    code = (
        "import sys; "
        "from unittest.mock import patch; "
        "import coola.utils.imports.torch_numpy as m; "
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
