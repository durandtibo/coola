from __future__ import annotations

import pytest

from tests.integration.helpers import run_python

RICH_CALLS = [
    "configure_rich_logging()",
    "make_progressbar()",
    "make_spinner()",
    "print_markdown('**a**')",
    "print_pretty({'a': 1})",
]


def missing_library_result(library: str, statement: str) -> list[str]:
    r"""Run a statement in a fresh interpreter where a library cannot be
    imported and return the words it prints."""
    code = (
        f"import sys; sys.modules[{library!r}] = None; import coola.display.rich as r; {statement}"
    )
    return run_python(code)


##################################################
#     Tests for the imports of coola.display     #
##################################################


def test_importing_display_does_not_import_the_optional_libraries() -> None:
    """Test that importing ``coola.display`` and ``coola.display.rich``
    imports neither ``rich``, ``pydantic`` nor ``colorlog``."""
    code = (
        "import sys; import coola.display; import coola.display.rich; "
        "print(*[n for n in ('rich', 'pydantic', 'colorlog') if n in sys.modules])"
    )
    assert run_python(code) == []


def test_importing_a_rich_name_imports_only_its_module() -> None:
    """Test that using a name of ``coola.display.rich`` imports the
    module that defines it."""
    code = (
        "import sys; from coola.display.rich import print_pretty; "
        "print('coola.display.rich.print_' in sys.modules, "
        "'coola.display.rich.progressbar' in sys.modules)"
    )
    assert run_python(code) == ["True", "False"]


##########################################
#     Tests without the rich package     #
##########################################


def test_importing_rich_display_without_rich_works() -> None:
    """Test that ``coola.display.rich`` can be imported when ``rich`` is
    not installed."""
    assert missing_library_result("rich", "print('ok')") == ["ok"]


@pytest.mark.parametrize("call", RICH_CALLS)
def test_rich_function_without_rich_raises_missing_package_error(call: str) -> None:
    """Test that a function that needs ``rich`` raises
    ``MissingPackageError`` with the install command."""
    statement = (
        "from coola.utils.imports import MissingPackageError\n"
        f"try:\n    r.{call}\nexcept MissingPackageError as e:\n"
        "    print('pip install rich' in str(e))"
    )
    assert missing_library_result("rich", statement) == ["True"]


def test_configure_rich_logging_without_rich_keeps_root_handlers() -> None:
    """Test that ``configure_rich_logging`` does not change the root
    logger when ``rich`` is not installed."""
    statement = (
        "import logging; before = list(logging.getLogger().handlers); "
        "from coola.utils.imports import MissingPackageError\n"
        "try:\n    r.configure_rich_logging(force=True)\nexcept MissingPackageError:\n    pass\n"
        "print(logging.getLogger().handlers == before)"
    )
    assert missing_library_result("rich", statement) == ["True"]


############################################
#     Tests without the colorlog package   #
############################################


def test_configure_colorlog_logging_without_colorlog_uses_plain_logging() -> None:
    """Test that ``configure_colorlog_logging`` configures plain logging
    when ``colorlog`` is not installed."""
    code = (
        "import sys; sys.modules['colorlog'] = None; import logging; "
        "from coola.display.colorlog import configure_colorlog_logging; "
        "configure_colorlog_logging(force=True); "
        "print(type(logging.getLogger().handlers[0]).__name__)"
    )
    assert run_python(code) == ["StreamHandler"]
