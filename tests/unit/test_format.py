from pytest import raises

from coola.format import str_dict, str_indent

##############################
#     Tests for str_dict     #
##############################


def test_str_dict_empty():
    assert str_dict(data={}) == ""


def test_str_dict_1_value():
    assert str_dict(data={"my_key": "my_value"}) == "my_key : my_value"


def test_str_dict_sorted_key():
    assert (
        str_dict(data={"short": 123, "long_key": 2}, sorted_keys=True)
        == "long_key : 2\nshort    : 123"
    )


def test_str_dict_unsorted_key():
    assert (
        str_dict(data={"short": 123, "long_key": 2}, sorted_keys=False)
        == "short    : 123\nlong_key : 2"
    )


def test_str_dict_nested_dict():
    assert str_dict(data={"my_key": {"my_key2": 123}}) == "my_key : {'my_key2': 123}"


def test_str_dict_incorrect_indent():
    with raises(ValueError):
        str_dict(data={"my_key": "my_value"}, indent=-1)


def test_str_dict_indent_2():
    assert str_dict(data={"my_key": "my_value"}, indent=2) == "  my_key : my_value"


################################
#     Tests for str_indent     #
################################


def test_str_indent_1_line():
    assert str_indent("abc") == "abc"


def test_str_indent_2_lines():
    assert str_indent("abc\n  def") == "abc\n    def"


def test_str_indent_num_spaces_2():
    assert str_indent("abc\ndef", num_spaces=2) == "abc\n  def"


def test_str_indent_num_spaces_4():
    assert str_indent("abc\ndef", num_spaces=4) == "abc\n    def"


def test_str_indent_not_a_string():
    assert str_indent(123) == "123"
