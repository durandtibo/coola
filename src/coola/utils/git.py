r"""Implement some utility functions to retrieve git repository
information."""

from __future__ import annotations

__all__ = [
    "get_current_branch",
    "get_current_commit_hash",
    "is_git_repo",
    "run_git_command",
]

import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_git_command(args: list[str], cwd: Path | str | None = None) -> str:
    r"""Run a git command and return its stripped stdout.

    Args:
        args: The git command arguments, excluding the ``git``
            executable itself.
        cwd: The working directory where the command is executed.
            If ``None``, the current working directory is used.

    Returns:
        The stripped standard output of the command.

    Raises:
        OSError: If the git command fails or ``git`` is not
            available.

    Example:
        ```pycon
        >>> from coola.utils.git import run_git_command
        >>> run_git_command(["rev-parse", "HEAD"])  # doctest: +SKIP

        ```

    Warning:
        ``args`` is passed to ``git`` without going through a shell,
        so it is not vulnerable to shell injection. However, the
        caller is still responsible for not passing untrusted input
        as ``args``, since arbitrary git subcommands and options
        (e.g. ones that read or write files) could be executed.
    """
    git_executable = shutil.which("git")
    if git_executable is None:
        msg = "Failed to run git command: the 'git' executable could not be found"
        raise OSError(msg)

    try:
        # The executable is resolved to an absolute path via `shutil.which`
        # above, and the arguments are git subcommands, not shell input.
        result = subprocess.run(  # noqa: S603
            [git_executable, *args],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        msg = f"Failed to run git command 'git {' '.join(args)}': {error}"
        raise OSError(msg) from error
    return result.stdout.strip()


def get_current_branch(cwd: Path | str | None = None) -> str:
    r"""Get the name of the current git branch.

    Args:
        cwd: The working directory where the command is executed.
            If ``None``, the current working directory is used.

    Returns:
        The name of the current branch.

    Raises:
        OSError: If the current directory is not part of a git
            repository or the branch name cannot be resolved.

    Example:
        ```pycon
        >>> from coola.utils.git import get_current_branch
        >>> get_current_branch()  # doctest: +SKIP

        ```
    """
    return run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)


def get_current_commit_hash(cwd: Path | str | None = None, short: bool = False) -> str:
    r"""Get the hash of the current git commit.

    Args:
        cwd: The working directory where the command is executed.
            If ``None``, the current working directory is used.
        short: If ``True``, returns the abbreviated commit hash.
            Defaults to ``False``.

    Returns:
        The hash of the current commit.

    Raises:
        OSError: If the current directory is not part of a git
            repository or the commit hash cannot be resolved.

    Example:
        ```pycon
        >>> from coola.utils.git import get_current_commit_hash
        >>> get_current_commit_hash()  # doctest: +SKIP

        ```
    """
    args = ["rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")
    return run_git_command(args, cwd=cwd)


def is_git_repo(cwd: Path | str | None = None) -> bool:
    r"""Indicate if the given directory is part of a git repository.

    Args:
        cwd: The working directory to check. If ``None``, the
            current working directory is used.

    Returns:
        ``True`` if the directory is part of a git repository,
            otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.git import is_git_repo
        >>> is_git_repo()  # doctest: +SKIP

        ```
    """
    try:
        run_git_command(["rev-parse", "--is-inside-work-tree"], cwd=cwd)
    except OSError:
        return False
    return True
