from __future__ import annotations

from coola.display.rich import make_progressbar, make_spinner
from coola.testing.marks import rich_available
from coola.utils.imports import is_rich_available

if is_rich_available():
    from rich import get_console
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

pytestmark = rich_available

######################################
#     Tests for make_progressbar     #
######################################


def test_make_progressbar_returns_progress_instance() -> None:
    assert isinstance(make_progressbar(), Progress)


def test_make_progressbar_default_transient_is_false() -> None:
    assert make_progressbar().live.transient is False


def test_make_progressbar_console() -> None:
    console = Console()
    assert make_progressbar(console=console).console is console


def test_make_progressbar_default_console_is_shared_console() -> None:
    assert make_progressbar().console is get_console()


def test_make_progressbar_transient_true() -> None:
    assert make_progressbar(transient=True).live.transient is True


def test_make_progressbar_column_types() -> None:
    progress = make_progressbar()
    column_types = [type(c) for c in progress.columns]
    assert column_types == [
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    ]


def test_make_progressbar_each_call_returns_new_instance() -> None:
    assert make_progressbar() is not make_progressbar()


##################################
#     Tests for make_spinner     #
##################################


def test_make_spinner_returns_progress_instance() -> None:
    assert isinstance(make_spinner(), Progress)


def test_make_spinner_default_transient_is_true() -> None:
    assert make_spinner().live.transient is True


def test_make_spinner_transient_false() -> None:
    assert make_spinner(transient=False).live.transient is False


def test_make_spinner_column_types() -> None:
    progress = make_spinner()
    column_types = [type(c) for c in progress.columns]
    assert column_types == [
        SpinnerColumn,
        TextColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
    ]


def test_make_spinner_each_call_returns_new_instance() -> None:
    assert make_spinner() is not make_spinner()


def test_make_spinner_console() -> None:
    console = Console()
    assert make_spinner(console=console).console is console


def test_make_spinner_default_console_is_shared_console() -> None:
    assert make_spinner().console is get_console()
