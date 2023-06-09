from __future__ import annotations

from pytest import raises

from coola.utils.format import str_dict, str_indent

##############################
#     Tests for str_dict     #
##############################


def test_str_dict_empty() -> None:
    assert str_dict(data={}) == ""


def test_str_dict_1_value() -> None:
    assert str_dict(data={"my_key": "my_value"}) == "my_key : my_value"


def test_str_dict_sorted_key() -> None:
    assert (
        str_dict(data={"short": 123, "long_key": 2}, sorted_keys=True)
        == "long_key : 2\nshort    : 123"
    )


def test_str_dict_unsorted_key() -> None:
    assert (
        str_dict(data={"short": 123, "long_key": 2}, sorted_keys=False)
        == "short    : 123\nlong_key : 2"
    )


def test_str_dict_nested_dict() -> None:
    assert str_dict(data={"my_key": {"my_key2": 123}}) == "my_key : {'my_key2': 123}"


def test_str_dict_incorrect_indent() -> None:
    with raises(ValueError, match="The indent has to be greater or equal to 0"):
        str_dict(data={"my_key": "my_value"}, indent=-1)


def test_str_dict_indent_2() -> None:
    assert str_dict(data={"my_key": "my_value"}, indent=2) == "  my_key : my_value"


################################
#     Tests for str_indent     #
################################


def test_str_indent_1_line() -> None:
    assert str_indent("abc") == "abc"


def test_str_indent_2_lines() -> None:
    assert str_indent("abc\n  def") == "abc\n    def"


def test_str_indent_num_spaces_2() -> None:
    assert str_indent("abc\ndef", num_spaces=2) == "abc\n  def"


def test_str_indent_num_spaces_4() -> None:
    assert str_indent("abc\ndef", num_spaces=4) == "abc\n    def"


def test_str_indent_not_a_string() -> None:
    assert str_indent(123) == "123"
