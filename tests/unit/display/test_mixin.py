"""Unit tests for MultilineDisplayMixin and InlineDisplayMixin."""

from __future__ import annotations

from typing import Any

import pytest

from coola.display import (
    BaseDisplayMixin,
    InlineDisplayMixin,
    MultilineDisplayMixin,
)
from coola.display.mixin import BaseDisplayMixin as BaseDisplayMixinFromSubmodule

# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------


class MultilineObj(MultilineDisplayMixin):
    def __init__(self, key1: str, key2: str) -> None:
        self.key1 = key1
        self.key2 = key2

    def _display_fields(self) -> dict[str, Any]:
        return {"key1": self.key1, "key2": self.key2}


class InlineObj(InlineDisplayMixin):
    def __init__(self, key1: str, key2: str) -> None:
        self.key1 = key1
        self.key2 = key2

    def _display_fields(self) -> dict[str, Any]:
        return {"key1": self.key1, "key2": self.key2}


class EmptyKwargsMultiline(MultilineDisplayMixin):
    def _display_fields(self) -> dict[str, Any]:
        return {}


class EmptyKwargsInline(InlineDisplayMixin):
    def _display_fields(self) -> dict[str, Any]:
        return {}


###########################################
#     Tests for MultilineDisplayMixin     #
###########################################


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


########################################
#     Tests for InlineDisplayMixin     #
########################################


def test_inline_repr() -> None:
    assert repr(InlineObj(key1="v1", key2="v2")) == "InlineObj(key1='v1', key2='v2')"


def test_inline_str() -> None:
    assert str(InlineObj(key1="v1", key2="v2")) == "InlineObj(key1=v1, key2=v2)"


def test_inline_repr_empty_kwargs() -> None:
    assert repr(EmptyKwargsInline()) == "EmptyKwargsInline()"


def test_inline_str_empty_kwargs() -> None:
    assert str(EmptyKwargsInline()) == "EmptyKwargsInline()"


######################################
#     Tests for BaseDisplayMixin     #
######################################


def test_base_display_mixin_exported_from_coola_display() -> None:
    assert BaseDisplayMixin is BaseDisplayMixinFromSubmodule


def test_base_display_mixin_is_base_of_multiline_and_inline_multiline() -> None:
    assert issubclass(MultilineDisplayMixin, BaseDisplayMixin)


def test_base_display_mixin_is_base_of_multiline_and_inline_inline() -> None:
    assert issubclass(InlineDisplayMixin, BaseDisplayMixin)


def test_base_display_mixin_custom_layout() -> None:
    class BracketObj(BaseDisplayMixin):
        def _display_fields(self) -> dict[str, Any]:
            return {"a": "x"}

        def _render_fields(self, fields: dict[str, Any], *, use_repr: bool) -> str:
            return f"{fields['a']!r}" if use_repr else f"{fields['a']!s}"

    assert repr(BracketObj()) == "'x'"


def test_base_display_mixin_abstract_raises() -> None:
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseDisplayMixin"):
        BaseDisplayMixin()  # ty: ignore[call-non-callable]


####################################
#     Tests for default fields     #
####################################


class NoFieldsInlineObj(InlineDisplayMixin):
    pass


class NoFieldsMultilineObj(MultilineDisplayMixin):
    pass


def test_display_fields_default_is_empty_dict() -> None:
    assert NoFieldsInlineObj()._display_fields() == {}


def test_no_fields_inline_repr() -> None:
    assert repr(NoFieldsInlineObj()) == "NoFieldsInlineObj()"


def test_no_fields_inline_str() -> None:
    assert str(NoFieldsInlineObj()) == "NoFieldsInlineObj()"


def test_no_fields_multiline_repr() -> None:
    assert repr(NoFieldsMultilineObj()) == "NoFieldsMultilineObj(\n  \n)"


def test_no_fields_multiline_str() -> None:
    assert str(NoFieldsMultilineObj()) == "NoFieldsMultilineObj(\n  \n)"
