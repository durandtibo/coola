from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.formatting.structure import (
    repr_indent,
    repr_mapping,
    repr_mapping_line,
    repr_sequence,
    repr_sequence_line,
    str_indent,
    str_mapping,
    str_mapping_line,
    str_sequence,
    str_sequence_line,
)

if TYPE_CHECKING:
    from collections.abc import Callable

#################################
#     Tests for repr_indent     #
#################################


def test_repr_indent_1_line() -> None:
    assert repr_indent("abc") == "abc"


def test_repr_indent_2_lines() -> None:
    assert repr_indent("abc\n  def") == "abc\n    def"


def test_repr_indent_num_spaces_0() -> None:
    assert repr_indent("abc\ndef", num_spaces=0) == "abc\ndef"


def test_repr_indent_num_spaces_2() -> None:
    assert repr_indent("abc\ndef") == "abc\n  def"


def test_repr_indent_num_spaces_4() -> None:
    assert repr_indent("abc\ndef", num_spaces=4) == "abc\n    def"


def test_repr_indent_num_spaces_incorrect() -> None:
    with pytest.raises(ValueError, match=r"num_spaces must be greater than or equal to 0, got -1"):
        repr_indent("abc\ndef", num_spaces=-1)


def test_repr_indent_not_a_repring() -> None:
    assert repr_indent(123) == "123"


##################################
#     Tests for repr_mapping     #
##################################


def test_repr_mapping_empty() -> None:
    assert repr_mapping({}) == ""


def test_repr_mapping_1_item() -> None:
    assert repr_mapping({"key": "value"}) == "(key): value"


def test_repr_mapping_2_items() -> None:
    assert repr_mapping({"key1": "value1", "key2": "value2"}) == "(key1): value1\n(key2): value2"


def test_repr_mapping_sorted_keys_true() -> None:
    assert (
        repr_mapping({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "(key1): value1\n(key2): value2"
    )


def test_repr_mapping_sorted_keys_false() -> None:
    assert repr_mapping({"key2": "value2", "key1": "value1"}) == "(key2): value2\n(key1): value1"


#######################################
#     Tests for repr_mapping_line     #
#######################################


def test_repr_mapping_line_empty() -> None:
    assert repr_mapping_line({}) == ""


def test_repr_mapping_line_1_item() -> None:
    assert repr_mapping_line({"key": "value"}) == "key='value'"


def test_repr_mapping_line_2_items() -> None:
    assert repr_mapping_line({"key1": "value1", "key2": "value2"}) == "key1='value1', key2='value2'"


def test_repr_mapping_line_sorted_keys_true() -> None:
    assert (
        repr_mapping_line({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "key1='value1', key2='value2'"
    )


def test_repr_mapping_line_sorted_keys_false() -> None:
    assert repr_mapping_line({"key2": "value2", "key1": "value1"}) == "key2='value2', key1='value1'"


def test_repr_mapping_line_separator() -> None:
    assert (
        repr_mapping_line({"key1": "abc", "key2": "meow", "key3": 42}, separator=" ")
        == "key1='abc' key2='meow' key3=42"
    )


###################################
#     Tests for repr_sequence     #
###################################


def test_repr_sequence_empty() -> None:
    assert repr_sequence([]) == ""


def test_repr_sequence_1_item() -> None:
    assert repr_sequence(["abc"]) == "(0): abc"


def test_repr_sequence_2_items() -> None:
    assert repr_sequence(["abc", 123]) == "(0): abc\n(1): 123"


def test_repr_sequence_2_items_multiple_line() -> None:
    assert repr_sequence(["abc", "something\nelse"]) == "(0): abc\n(1): something\n  else"


########################################
#     Tests for repr_sequence_line     #
########################################


def test_repr_sequence_line_empty() -> None:
    assert repr_sequence_line([]) == ""


def test_repr_sequence_line_1_item() -> None:
    assert repr_sequence_line(["abc"]) == "'abc'"


def test_repr_sequence_line_2_items() -> None:
    assert repr_sequence_line(["abc", 123]) == "'abc', 123"


def test_repr_sequence_line_2_items_separator() -> None:
    assert repr_sequence_line(["abc", "meow", 42], separator="|") == "'abc'|'meow'|42"


################################
#     Tests for str_indent     #
################################


def test_str_indent_1_line() -> None:
    assert str_indent("abc") == "abc"


def test_str_indent_2_lines() -> None:
    assert str_indent("abc\n  def") == "abc\n    def"


def test_str_indent_num_spaces_0() -> None:
    assert str_indent("abc\ndef", num_spaces=0) == "abc\ndef"


def test_str_indent_num_spaces_2() -> None:
    assert str_indent("abc\ndef") == "abc\n  def"


def test_str_indent_num_spaces_4() -> None:
    assert str_indent("abc\ndef", num_spaces=4) == "abc\n    def"


def test_str_indent_num_spaces_incorrect() -> None:
    with pytest.raises(ValueError, match=r"num_spaces must be greater than or equal to 0, got -1"):
        str_indent("abc\ndef", num_spaces=-1)


def test_str_indent_not_a_string() -> None:
    assert str_indent(123) == "123"


#################################
#     Tests for str_mapping     #
#################################


def test_str_mapping_empty() -> None:
    assert str_mapping({}) == ""


def test_str_mapping_1_item() -> None:
    assert str_mapping({"key": "value"}) == "(key): value"


def test_str_mapping_2_items() -> None:
    assert str_mapping({"key1": "value1", "key2": "value2"}) == "(key1): value1\n(key2): value2"


def test_str_mapping_sorted_keys_true() -> None:
    assert (
        str_mapping({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "(key1): value1\n(key2): value2"
    )


def test_str_mapping_sorted_keys_false() -> None:
    assert str_mapping({"key2": "value2", "key1": "value1"}) == "(key2): value2\n(key1): value1"


######################################
#     Tests for str_mapping_line     #
######################################


def test_str_mapping_line_empty() -> None:
    assert str_mapping_line({}) == ""


def test_str_mapping_line_1_item() -> None:
    assert str_mapping_line({"key": "value"}) == "key=value"


def test_str_mapping_line_2_items() -> None:
    assert str_mapping_line({"key1": "value1", "key2": "value2"}) == "key1=value1, key2=value2"


def test_str_mapping_line_sorted_keys_true() -> None:
    assert (
        str_mapping_line({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "key1=value1, key2=value2"
    )


def test_str_mapping_line_sorted_keys_false() -> None:
    assert str_mapping_line({"key2": "value2", "key1": "value1"}) == "key2=value2, key1=value1"


def test_str_mapping_line_separator() -> None:
    assert (
        str_mapping_line({"key1": "abc", "key2": "meow", "key3": 42}, separator=" ")
        == "key1=abc key2=meow key3=42"
    )


##################################
#     Tests for str_sequence     #
##################################


def test_str_sequence_empty() -> None:
    assert str_sequence([]) == ""


def test_str_sequence_1_item() -> None:
    assert str_sequence(["abc"]) == "(0): abc"


def test_str_sequence_2_items() -> None:
    assert str_sequence(["abc", 123]) == "(0): abc\n(1): 123"


def test_str_sequence_2_items_multiple_line() -> None:
    assert str_sequence(["abc", "something\nelse"]) == "(0): abc\n(1): something\n  else"


#######################################
#     Tests for str_sequence_line     #
#######################################


def test_str_sequence_line_empty() -> None:
    assert str_sequence_line([]) == ""


def test_str_sequence_line_1_item() -> None:
    assert str_sequence_line(["abc"]) == "abc"


def test_str_sequence_line_2_items() -> None:
    assert str_sequence_line(["abc", 123]) == "abc, 123"


def test_str_sequence_line_2_items_separator() -> None:
    assert str_sequence_line(["abc", "meow", 42], separator="|") == "abc|meow|42"


def test_str_indent_skips_trailing_empty_line() -> None:
    assert str_indent("a\nb\n", num_spaces=2) == "a\n  b\n"


def test_str_indent_skips_empty_lines() -> None:
    assert str_indent("a\n\nb", num_spaces=2) == "a\n\n  b"


def test_repr_mapping_sorted_keys_mixed_key_types() -> None:
    assert repr_mapping({"b": 1, 1: 2}, sorted_keys=True) == "(1): 2\n(b): 1"


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (repr_indent, ("a", 2)),
        (str_indent, ("a", 2)),
        (repr_mapping, ({"a": 1}, True)),
        (str_mapping, ({"a": 1}, True)),
        (repr_mapping_line, ({"a": 1}, True)),
        (str_mapping_line, ({"a": 1}, True)),
        (repr_sequence, ([1], 2)),
        (str_sequence, ([1], 2)),
        (repr_sequence_line, ([1], ", ")),
        (str_sequence_line, ([1], ", ")),
    ],
)
def test_options_are_keyword_only(function: Callable, args: tuple) -> None:
    with pytest.raises(TypeError, match="positional argument"):
        function(*args)
