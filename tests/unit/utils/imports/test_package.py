from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    MissingPackageError,
    check_package,
    module_available,
    package_available,
    raise_package_missing_error,
)

logger = logging.getLogger(__name__)


MODULE = "coola.utils.imports.package"


#######################################
#     Tests for package_available     #
#######################################


def test_package_available_true() -> None:
    assert package_available("os")


def test_package_available_false() -> None:
    assert not package_available("missing_package")


@pytest.mark.parametrize(
    ("error", "name"), [(ValueError, "pkg_value_error"), (ImportError, "pkg_import_error")]
)
def test_package_available_false_if_find_spec_fails(error: type[Exception], name: str) -> None:
    # The name differs per case because the result is cached.
    with patch(f"{MODULE}.find_spec", side_effect=error("failed")):
        assert not package_available(name)


def test_package_available_false_subpackage() -> None:
    assert not package_available("missing.package")


######################################
#     Tests for module_available     #
######################################


def test_module_available_true() -> None:
    assert module_available("os")


def test_module_available_false() -> None:
    assert not module_available("os.missing")


def test_module_available_false_submodule() -> None:
    assert not module_available("missing.module")


###################################
#     Tests for check_package     #
###################################


def test_check_package_exist() -> None:
    check_package("os")


def test_check_package_missing() -> None:
    with pytest.raises(
        MissingPackageError, match=r"'missing_package' package is required but not installed."
    ):
        check_package("missing_package")


def test_check_package_missing_message() -> None:
    msg = (
        "'missing_package' package is required but not installed. "
        "You can install 'missing_package' package with the command:\n\n"
        "pip install missing_package\n\nor\n\n"
        "uv pip install missing_package\n"
    )
    with pytest.raises(MissingPackageError, match="'missing_package' package is required") as info:
        check_package("missing_package")
    assert str(info.value) == msg


def test_check_package_missing_message_with_install() -> None:
    msg = (
        "'missing_package' package is required but not installed. "
        "You can install 'missing_package' package with the command:\n\n"
        "pip install my-package\n\nor\n\n"
        "uv pip install my-package\n"
    )
    with pytest.raises(MissingPackageError, match="'missing_package' package is required") as info:
        check_package("missing_package", install="my-package")
    assert str(info.value) == msg


def test_check_package_message_is_the_one_of_raise_package_missing_error() -> None:
    with pytest.raises(
        MissingPackageError, match="'missing_package' package is required"
    ) as check_info:
        check_package("missing_package")
    with pytest.raises(
        MissingPackageError, match="'missing_package' package is required"
    ) as raise_info:
        raise_package_missing_error("missing_package")
    assert str(check_info.value) == str(raise_info.value)


def test_check_package_install_is_keyword_only() -> None:
    function: Any = check_package
    with pytest.raises(TypeError, match="positional argument"):
        function("os", "pip install os")


def test_check_package_imports_the_package() -> None:
    with (
        patch(f"{MODULE}.importlib.import_module", side_effect=OSError("broken")),
        pytest.raises(MissingPackageError, match="'broken_package' package is required"),
    ):
        check_package("broken_package")


#################################################
#     Tests for raise_package_missing_error     #
#################################################


def test_raise_package_missing_error_basic() -> None:
    with pytest.raises(
        MissingPackageError, match=r"'mypackage' package is required but not installed."
    ):
        raise_package_missing_error("mypackage")


def test_raise_package_missing_error_default_install() -> None:
    msg = (
        "'testpkg' package is required but not installed. "
        "You can install 'testpkg' package with the command:\n\n"
        "pip install testpkg\n\nor\n\n"
        "uv pip install testpkg\n"
    )
    with pytest.raises(MissingPackageError, match="'testpkg' package is required") as info:
        raise_package_missing_error("testpkg")
    assert str(info.value) == msg


def test_raise_package_missing_error_install() -> None:
    msg = (
        "'testpkg' package is required but not installed. "
        "You can install 'testpkg' package with the command:\n\n"
        "pip install test-package\n\nor\n\n"
        "uv pip install test-package\n"
    )
    with pytest.raises(MissingPackageError, match="'testpkg' package is required") as info:
        raise_package_missing_error("testpkg", install="test-package")
    assert str(info.value) == msg


#########################################
#     Tests for MissingPackageError     #
#########################################


def test_missing_package_error_is_module_not_found_error() -> None:
    assert issubclass(MissingPackageError, ModuleNotFoundError)


def test_missing_package_error_is_import_error() -> None:
    assert issubclass(MissingPackageError, ImportError)


def test_missing_package_error_is_runtime_error() -> None:
    assert issubclass(MissingPackageError, RuntimeError)


def test_check_package_missing_raises_missing_package_error() -> None:
    with pytest.raises(MissingPackageError, match="'not_a_package' package is required"):
        check_package("not_a_package")


def test_check_package_missing_error_name() -> None:
    with pytest.raises(MissingPackageError, match="'not_a_package' package is required") as info:
        check_package("not_a_package")
    assert info.value.name == "not_a_package"


def test_raise_package_missing_error_raises_missing_package_error() -> None:
    with pytest.raises(MissingPackageError, match="'my_package' package is required"):
        raise_package_missing_error("my_package", install="my-package")


def test_raise_package_missing_error_name() -> None:
    with pytest.raises(MissingPackageError, match="'my_package' package is required") as info:
        raise_package_missing_error("my_package", install="my-package")
    assert info.value.name == "my_package"


def test_missing_package_error_can_be_caught_as_import_error() -> None:
    with pytest.raises(ImportError, match="package is required"):
        check_package("not_a_package")


def test_missing_package_error_can_still_be_caught_as_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="package is required"):
        check_package("not_a_package")


##########################################################
#     Tests for module_available with broken modules     #
##########################################################


BROKEN_ERRORS = [OSError("libfoo.so not found"), RuntimeError("ABI"), AttributeError("x")]


@pytest.mark.parametrize("error", BROKEN_ERRORS)
def test_module_available_broken_module_returns_false(error: Exception) -> None:
    name = f"broken_module_{type(error).__name__}"
    with patch(f"{MODULE}.importlib.import_module", side_effect=error):
        assert not module_available(name)


@pytest.mark.parametrize("error", BROKEN_ERRORS)
def test_module_available_broken_module_warns(
    error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    name = f"broken_module_warns_{type(error).__name__}"
    with (
        patch(f"{MODULE}.importlib.import_module", side_effect=error),
        caplog.at_level(logging.WARNING, logger=MODULE),
    ):
        module_available(name)
    assert f"The module '{name}' is installed but cannot be imported" in caplog.text


def test_module_available_missing_module_returns_false() -> None:
    assert not module_available("missing_module_returns_false")


def test_module_available_missing_module_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=MODULE):
        module_available("missing_module_without_warning")
    assert not caplog.records


def test_package_available_with_a_broken_package() -> None:
    with patch(f"{MODULE}.importlib.import_module", side_effect=OSError("broken")):
        assert package_available("json")


def test_package_available_does_not_import_the_package() -> None:
    with patch(f"{MODULE}.importlib.import_module", side_effect=OSError("broken")) as mock:
        package_available("json")
    mock.assert_not_called()
