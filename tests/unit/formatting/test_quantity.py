from __future__ import annotations

import pytest

from coola.formatting.quantity import (
    find_best_byte_unit,
    make_bar,
    str_human_byte_size,
    str_time_human,
)

##############################
#     Tests for make_bar     #
##############################


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_make_bar_raises_value_error(value: float) -> None:
    with pytest.raises(ValueError, match="value must be in"):
        make_bar(value)


def test_make_bar_raises_value_error_if_length_is_zero() -> None:
    with pytest.raises(ValueError, match="length must be greater than 0"):
        make_bar(0.5, length=0)


def test_make_bar_raises_value_error_if_length_is_negative() -> None:
    with pytest.raises(ValueError, match="length must be greater than 0"):
        make_bar(0.5, length=-1)


def test_make_bar_empty() -> None:
    assert make_bar(0.0) == "[░░░░░░░░░░]"


def test_make_bar_full() -> None:
    assert make_bar(1.0) == "[██████████]"


def test_make_bar_half() -> None:
    assert make_bar(0.5) == "[█████░░░░░]"


def test_make_bar_partial() -> None:
    assert make_bar(0.6) == "[██████░░░░]"


def test_make_bar_low_value() -> None:
    assert make_bar(0.1) == "[█░░░░░░░░░]"


def test_make_bar_int_value_zero() -> None:
    assert make_bar(0) == "[░░░░░░░░░░]"


def test_make_bar_int_value_one() -> None:
    assert make_bar(1) == "[██████████]"


def test_make_bar_custom_length_short() -> None:
    assert make_bar(0.5, length=4) == "[██░░]"


def test_make_bar_custom_length_long() -> None:
    assert make_bar(0.5, length=20) == "[██████████░░░░░░░░░░]"


def test_make_bar_length_one_empty() -> None:
    assert make_bar(0.0, length=1) == "[░]"


def test_make_bar_length_one_full() -> None:
    assert make_bar(1.0, length=1) == "[█]"


def test_make_bar_starts_with_open_bracket() -> None:
    assert make_bar(0.5).startswith("[")


def test_make_bar_ends_with_close_bracket() -> None:
    assert make_bar(0.5).endswith("]")


def test_make_bar_total_length_matches_length_param() -> None:
    bar = make_bar(0.5, length=10)
    # Strip brackets and count characters
    inner = bar[1:-1]
    assert len(inner) == 10


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


def test_str_human_byte_size_zero() -> None:
    """Test str_human_byte_size with size of zero."""
    assert str_human_byte_size(0) == "0.00 B"


def test_str_human_byte_size_exact_kb_boundary() -> None:
    """Test str_human_byte_size at exact KB boundary (1024 bytes).

    Note: At exactly 1024 bytes, 1024/1024 = 1.0, which upgrades the
    unit to 'KB'.
    """
    assert str_human_byte_size(1024) == "1.00 KB"


def test_str_human_byte_size_just_over_kb() -> None:
    """Test str_human_byte_size just over KB boundary (1025 bytes)."""
    assert str_human_byte_size(1025) == "1.00 KB"


def test_str_human_byte_size_just_under_kb() -> None:
    """Test str_human_byte_size just under KB boundary (1023 bytes)."""
    assert str_human_byte_size(1023) == "1,023.00 B"


def test_str_human_byte_size_exact_mb_boundary() -> None:
    """Test str_human_byte_size at exact MB boundary (1048576 bytes).

    Note: At exactly 1048576 bytes, 1048576/1048576 = 1.0, which upgrades
    the unit to 'MB'.
    """
    assert str_human_byte_size(1048576) == "1.00 MB"


def test_str_human_byte_size_just_over_mb() -> None:
    """Test str_human_byte_size just over MB boundary."""
    assert str_human_byte_size(1048577) == "1.00 MB"


def test_str_human_byte_size_exact_gb_boundary() -> None:
    """Test str_human_byte_size at exact GB boundary (1073741824 bytes).

    Note: At exactly 1073741824 bytes, the function upgrades the unit to
    'GB'.
    """
    assert str_human_byte_size(1073741824) == "1.00 GB"


def test_str_human_byte_size_incorrect_unit() -> None:
    with pytest.raises(ValueError, match=r"Incorrect unit ''. The available units are"):
        str_human_byte_size(1, "")


def test_str_human_byte_size_negative_size() -> None:
    with pytest.raises(ValueError, match=r"Size must be greater than or equal to 0"):
        str_human_byte_size(-1)


#########################################
#     Tests for find_best_byte_unit     #
#########################################


@pytest.mark.parametrize(("size", "unit"), [(2, "B"), (1023, "B"), (2048, "KB"), (2097152, "MB")])
def test_find_best_byte_unit(size: int, unit: str) -> None:
    assert find_best_byte_unit(size) == unit


def test_find_best_byte_unit_zero() -> None:
    """Test find_best_byte_unit with size of zero."""
    assert find_best_byte_unit(0) == "B"


def test_find_best_byte_unit_exact_kb_boundary() -> None:
    """Test find_best_byte_unit at exact KB boundary (1024 bytes).

    Note: At exactly 1024 bytes, 1024/1024 = 1.0, which upgrades the
    unit to 'KB'.
    """
    assert find_best_byte_unit(1024) == "KB"


def test_find_best_byte_unit_just_over_kb() -> None:
    """Test find_best_byte_unit just over KB boundary (1025 bytes)."""
    assert find_best_byte_unit(1025) == "KB"


def test_find_best_byte_unit_exact_mb_boundary() -> None:
    """Test find_best_byte_unit at exact MB boundary (1048576 bytes).

    Note: At exactly 1048576 bytes, the function returns 'MB'.
    """
    assert find_best_byte_unit(1048576) == "MB"


def test_find_best_byte_unit_just_over_mb() -> None:
    """Test find_best_byte_unit just over MB boundary."""
    assert find_best_byte_unit(1048577) == "MB"


def test_find_best_byte_unit_exact_gb_boundary() -> None:
    """Test find_best_byte_unit at exact GB boundary (1073741824 bytes).

    Note: At exactly 1073741824 bytes, the function returns 'GB'.
    """
    assert find_best_byte_unit(1073741824) == "GB"


def test_find_best_byte_unit_negative_size() -> None:
    with pytest.raises(ValueError, match=r"Size must be greater than or equal to 0"):
        find_best_byte_unit(-1)


def test_str_time_human_negative() -> None:
    assert str_time_human(-5) == "-0:00:05"


def test_str_time_human_negative_with_microseconds() -> None:
    assert str_time_human(-3661.2) == "-1:01:01.200000"


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), float("-inf"), 1e30])
def test_str_time_human_invalid(seconds: float) -> None:
    with pytest.raises(ValueError, match="seconds"):
        str_time_human(seconds)


def test_str_human_byte_size_rounds_up_to_the_next_unit() -> None:
    assert str_human_byte_size(1048575) == "1.00 MB"


def test_str_human_byte_size_largest_unit_is_not_promoted() -> None:
    assert str_human_byte_size(1024**4 * 2000) == "2,000.00 TB"
