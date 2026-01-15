from __future__ import annotations

import pytest

from coola.utils.format import (
    find_best_byte_unit,
    repr_indent,
    repr_mapping,
    repr_mapping_line,
    repr_sequence,
    repr_sequence_line,
    str_human_byte_size,
    str_indent,
    str_mapping,
    str_mapping_line,
    str_sequence,
    str_sequence_line,
    str_time_human,
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
    with pytest.raises(RuntimeError):
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
    with pytest.raises(RuntimeError):
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


####################################
#     Tests for str_time_human     #
####################################


@pytest.mark.parametrize(
    ("seconds", "human"),
    [
        (1, "0:00:01"),
        (61, "0:01:01"),
        (3661, "1:01:01"),
        (3661.0, "1:01:01"),
        (1.1, "0:00:01.100000"),
        (3600 * 24 + 3661, "1 day, 1:01:01"),
        (3600 * 48 + 3661, "2 days, 1:01:01"),
    ],
)
def test_str_time_human(seconds: float, human: str) -> None:
    assert str_time_human(seconds) == human


#########################################
#     Tests for str_human_byte_size     #
#########################################


@pytest.mark.parametrize(
    ("size", "output"), [(2, "2.00 B"), (2048, "2,048.00 B"), (2097152, "2,097,152.00 B")]
)
def test_str_human_byte_size_b(size: int, output: str) -> None:
    assert str_human_byte_size(size, "B") == output


@pytest.mark.parametrize(
    ("size", "output"),
    [(2048, "2.00 KB"), (2097152, "2,048.00 KB"), (2147483648, "2,097,152.00 KB")],
)
def test_str_human_byte_size_kb(size: int, output: str) -> None:
    assert str_human_byte_size(size, "KB") == output


@pytest.mark.parametrize(
    ("size", "output"),
    [(2048, "0.00 MB"), (2097152, "2.00 MB"), (2147483648, "2,048.00 MB")],
)
def test_str_human_byte_size_mb(size: int, output: str) -> None:
    assert str_human_byte_size(size, "MB") == output


@pytest.mark.parametrize(
    ("size", "output"), [(2048, "0.00 GB"), (2097152, "0.00 GB"), (2147483648, "2.00 GB")]
)
def test_str_human_byte_size_gb(size: int, output: str) -> None:
    assert str_human_byte_size(size, "GB") == output


@pytest.mark.parametrize(
    ("size", "output"),
    [(2, "2.00 B"), (1023, "1,023.00 B"), (2048, "2.00 KB"), (2097152, "2.00 MB")],
)
def test_str_human_byte_size_auto(size: int, output: str) -> None:
    assert str_human_byte_size(size) == output


def test_str_human_byte_size_incorrect_unit() -> None:
    with pytest.raises(ValueError, match=r"Incorrect unit ''. The available units are"):
        assert str_human_byte_size(1, "")


def test_str_human_byte_size_negative_size() -> None:
    with pytest.raises(ValueError, match=r"Size must be non-negative"):
        assert str_human_byte_size(-1)


#########################################
#     Tests for find_best_byte_unit     #
#########################################


@pytest.mark.parametrize(("size", "unit"), [(2, "B"), (1023, "B"), (2048, "KB"), (2097152, "MB")])
def test_find_best_byte_unit(size: int, unit: str) -> None:
    assert find_best_byte_unit(size) == unit


def test_find_best_byte_unit_negative_size() -> None:
    with pytest.raises(ValueError, match=r"Size must be non-negative"):
        assert find_best_byte_unit(-1)
