"""Unit tests for MultilineDisplayMixin and InlineDisplayMixin."""

from __future__ import annotations

from typing import Any

import pytest

from coola.display import BaseDisplayMixin, InlineDisplayMixin, MultilineDisplayMixin
from coola.display.mixin import BaseDisplayMixin as BaseDisplayMixinFromSubmodule

# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------


class MultilineObj(MultilineDisplayMixin):
    def __init__(self, key1: str, key2: str) -> None:
        self.key1 = key1
        self.key2 = key2

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"key1": self.key1, "key2": self.key2}


class InlineObj(InlineDisplayMixin):
    def __init__(self, key1: str, key2: str) -> None:
        self.key1 = key1
        self.key2 = key2

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"key1": self.key1, "key2": self.key2}


class EmptyKwargsMultiline(MultilineDisplayMixin):
    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {}


class EmptyKwargsInline(InlineDisplayMixin):
    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {}


##########################################
#     Tests for MultilineDisplayMixin    #
##########################################


def test_multiline_repr() -> None:
    assert repr(MultilineObj(key1="v1", key2="v2")) == (
        "MultilineObj(\n  (key1): v1\n  (key2): v2\n)"
    )


def test_multiline_str() -> None:
    assert str(MultilineObj(key1="v1", key2="v2")) == (
        "MultilineObj(\n  (key1): v1\n  (key2): v2\n)"
    )


def test_multiline_repr_empty_kwargs() -> None:
    assert repr(EmptyKwargsMultiline()) == "EmptyKwargsMultiline(\n  \n)"


def test_multiline_str_empty_kwargs() -> None:
    assert str(EmptyKwargsMultiline()) == "EmptyKwargsMultiline(\n  \n)"


def test_multiline_abstract_without_get_repr_kwargs_raises() -> None:
    with pytest.raises(TypeError, match="Can't instantiate abstract class MultilineDisplayMixin"):
        MultilineDisplayMixin()


#######################################
#     Tests for InlineDisplayMixin    #
#######################################


def test_inline_repr() -> None:
    assert repr(InlineObj(key1="v1", key2="v2")) == "InlineObj(key1='v1', key2='v2')"


def test_inline_str() -> None:
    assert str(InlineObj(key1="v1", key2="v2")) == "InlineObj(key1=v1, key2=v2)"


def test_inline_repr_empty_kwargs() -> None:
    assert repr(EmptyKwargsInline()) == "EmptyKwargsInline()"


def test_inline_str_empty_kwargs() -> None:
    assert str(EmptyKwargsInline()) == "EmptyKwargsInline()"


def test_inline_abstract_without_get_repr_kwargs_raises() -> None:
    with pytest.raises(TypeError, match="Can't instantiate abstract class InlineDisplayMixin"):
        InlineDisplayMixin()


#########################################
#     Tests for BaseDisplayMixin       #
#########################################


def test_base_display_mixin_exported_from_coola_display() -> None:
    assert BaseDisplayMixin is BaseDisplayMixinFromSubmodule


def test_base_display_mixin_is_base_of_multiline_and_inline() -> None:
    assert issubclass(MultilineDisplayMixin, BaseDisplayMixin)
    assert issubclass(InlineDisplayMixin, BaseDisplayMixin)


def test_base_display_mixin_abstract_raises() -> None:
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseDisplayMixin"):
        BaseDisplayMixin()
