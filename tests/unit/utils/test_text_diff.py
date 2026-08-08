"""Unit tests for zenpyre.utils.text_diff."""

from __future__ import annotations

import pytest

from coola.utils.text_diff import unified_diff

TEXT_A = "Cats are independent animals.\nThey sleep up to 16 hours a day.\nCats are carnivores.\n"
TEXT_B = (
    "Cats are independent animals.\n"
    "They sleep up to 20 hours a day.\n"
    "Cats are obligate carnivores.\n"
    "They are known for their agility.\n"
)

_DEFAULT_DIFF = (
    "--- previous\n"
    "+++ latest\n"
    "@@ -1,3 +1,4 @@\n"
    " Cats are independent animals.\n"
    "-They sleep up to 16 hours a day.\n"
    "-Cats are carnivores.\n"
    "+They sleep up to 20 hours a day.\n"
    "+Cats are obligate carnivores.\n"
    "+They are known for their agility.\n"
)


#################################
#     Tests for unified_diff    #
#################################


# --- Return type ---


def test_unified_diff_returns_str() -> None:
    assert isinstance(unified_diff(TEXT_A, TEXT_B), str)


# --- Identical / empty ---


def test_unified_diff_identical_texts() -> None:
    assert unified_diff(TEXT_A, TEXT_A) == ""


def test_unified_diff_both_empty() -> None:
    assert unified_diff("", "") == ""


def test_unified_diff_empty_before() -> None:
    assert unified_diff("", "Cats are cute.\n") == (
        "--- previous\n+++ latest\n@@ -0,0 +1 @@\n+Cats are cute.\n"
    )


def test_unified_diff_empty_after() -> None:
    assert unified_diff("Cats are cute.\n", "") == (
        "--- previous\n+++ latest\n@@ -1 +0,0 @@\n-Cats are cute.\n"
    )


# --- Diff content ---


def test_unified_diff_default() -> None:
    assert unified_diff(TEXT_A, TEXT_B) == _DEFAULT_DIFF


# --- Labels ---


def test_unified_diff_custom_label_a() -> None:
    assert unified_diff(TEXT_A, TEXT_B, label_a="old") == _DEFAULT_DIFF.replace(
        "--- previous", "--- old"
    )


def test_unified_diff_custom_label_b() -> None:
    assert unified_diff(TEXT_A, TEXT_B, label_b="new") == _DEFAULT_DIFF.replace(
        "+++ latest", "+++ new"
    )


# --- Context lines ---


def test_unified_diff_context_lines_default() -> None:
    a = "\n".join(str(i) for i in range(20)) + "\n"
    b = a.replace("10\n", "99\n")
    assert unified_diff(a, b) == (
        "--- previous\n+++ latest\n@@ -8,7 +8,7 @@\n 7\n 8\n 9\n-10\n+99\n 11\n 12\n 13\n"
    )


def test_unified_diff_context_lines_zero() -> None:
    a = "\n".join(str(i) for i in range(20)) + "\n"
    b = a.replace("10\n", "99\n")
    assert unified_diff(a, b, context_lines=0) == (
        "--- previous\n+++ latest\n@@ -11 +11 @@\n-10\n+99\n"
    )


def test_unified_diff_context_lines_one() -> None:
    a = "\n".join(str(i) for i in range(20)) + "\n"
    b = a.replace("10\n", "99\n")
    assert unified_diff(a, b, context_lines=1) == (
        "--- previous\n+++ latest\n@@ -10,3 +10,3 @@\n 9\n-10\n+99\n 11\n"
    )


def test_unified_diff_negative_context_lines_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        unified_diff(TEXT_A, TEXT_B, context_lines=-1)
