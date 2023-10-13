from __future__ import annotations

from pytest import raises

from coola.utils.format import (
    repr_indent,
    repr_mapping,
    repr_sequence,
    str_indent,
    str_mapping,
    str_sequence,
)

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
    with raises(RuntimeError):
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


def test_repr_mapping_sorted_values_true() -> None:
    assert (
        repr_mapping({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "(key1): value1\n(key2): value2"
    )


def test_repr_mapping_sorted_values_false() -> None:
    assert repr_mapping({"key2": "value2", "key1": "value1"}) == "(key2): value2\n(key1): value1"


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
    with raises(RuntimeError):
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


def test_str_mapping_sorted_values_true() -> None:
    assert (
        str_mapping({"key2": "value2", "key1": "value1"}, sorted_keys=True)
        == "(key1): value1\n(key2): value2"
    )


def test_str_mapping_sorted_values_false() -> None:
    assert str_mapping({"key2": "value2", "key1": "value1"}) == "(key2): value2\n(key1): value1"


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
