"""Unit tests for git utilities."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from coola.utils.git import (
    get_current_branch,
    get_current_commit_hash,
    is_git_repo,
    run_git_command,
)

GIT_PATH = "/usr/bin/git"

######################################
#     Tests for get_current_branch    #
######################################


def test_get_current_branch_returns_stdout() -> None:
    """Test that the branch name is returned from git stdout."""
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="main\n")
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch("subprocess.run", return_value=completed) as mock_run,
    ):
        assert get_current_branch() == "main"
        mock_run.assert_called_once_with(
            [GIT_PATH, "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=None,
            capture_output=True,
            check=True,
            text=True,
        )


def test_get_current_branch_raises_error_when_not_a_repo() -> None:
    """Test that an OSError is raised when git command fails."""
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ),
        pytest.raises(OSError, match="Failed to run git command"),
    ):
        get_current_branch()


############################################
#     Tests for get_current_commit_hash    #
############################################


def test_get_current_commit_hash_returns_stdout() -> None:
    """Test that the commit hash is returned from git stdout."""
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="abc1234\n")
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch("subprocess.run", return_value=completed) as mock_run,
    ):
        assert get_current_commit_hash() == "abc1234"
        mock_run.assert_called_once_with(
            [GIT_PATH, "rev-parse", "HEAD"],
            cwd=None,
            capture_output=True,
            check=True,
            text=True,
        )


def test_get_current_commit_hash_short() -> None:
    """Test that the short commit hash is requested when short=True."""
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="abc1234\n")
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch("subprocess.run", return_value=completed) as mock_run,
    ):
        assert get_current_commit_hash(short=True) == "abc1234"
        mock_run.assert_called_once_with(
            [GIT_PATH, "rev-parse", "--short", "HEAD"],
            cwd=None,
            capture_output=True,
            check=True,
            text=True,
        )


def test_get_current_commit_hash_raises_error_when_not_a_repo() -> None:
    """Test that an OSError is raised when git command fails."""
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ),
        pytest.raises(OSError, match="Failed to run git command"),
    ):
        get_current_commit_hash()


##################################
#     Tests for is_git_repo     #
##################################


def test_is_git_repo_returns_true() -> None:
    """Test that True is returned when inside a git repository."""
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="true\n")
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch("subprocess.run", return_value=completed),
    ):
        assert is_git_repo() is True


def test_is_git_repo_returns_false() -> None:
    """Test that False is returned when not inside a git repository."""
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ),
    ):
        assert is_git_repo() is False


def test_is_git_repo_returns_false_when_git_not_found() -> None:
    """Test that False is returned when the git executable is
    missing."""
    with patch("shutil.which", return_value=None):
        assert is_git_repo() is False


########################################
#     Tests for run_git_command       #
########################################


def test_run_git_command_returns_stripped_stdout() -> None:
    """Test that the stdout is stripped and returned."""
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="  hello  \n")
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch("subprocess.run", return_value=completed) as mock_run,
    ):
        assert run_git_command(["status"]) == "hello"
        mock_run.assert_called_once_with(
            [GIT_PATH, "status"],
            cwd=None,
            capture_output=True,
            check=True,
            text=True,
        )


def test_run_git_command_passes_cwd() -> None:
    """Test that the cwd argument is forwarded to subprocess.run."""
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n")
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch("subprocess.run", return_value=completed) as mock_run,
    ):
        assert run_git_command(["status"], cwd="/path/to/repo") == "ok"
        mock_run.assert_called_once_with(
            [GIT_PATH, "status"],
            cwd="/path/to/repo",
            capture_output=True,
            check=True,
            text=True,
        )


def test_run_git_command_raises_error_on_called_process_error() -> None:
    """Test that an OSError is raised when the git command fails."""
    with (
        patch("shutil.which", return_value=GIT_PATH),
        patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ),
        pytest.raises(OSError, match="Failed to run git command"),
    ):
        run_git_command(["status"])


def test_run_git_command_raises_error_when_git_not_found() -> None:
    """Test that an OSError is raised when the git executable is
    missing."""
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(OSError, match="git' executable could not be found"),
    ):
        run_git_command(["status"])
