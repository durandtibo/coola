from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from coola.identifier import generate_checksummed_id, verify_checksummed_id
from coola.identifier.checksummed import _checksum_symbol

URANDOM = "coola.identifier.checksummed.os.urandom"
DEFAULT_PATTERN = re.compile(
    r"^[0-9A-HJKMNP-TV-Z]{4}(-[0-9A-HJKMNP-TV-Z]{4}){2}-[0-9A-HJKMNP-TV-Z]$"
)

#############################################
#     Tests for generate_checksummed_id     #
#############################################


def test_generate_checksummed_id_returns_str() -> None:
    assert isinstance(generate_checksummed_id(), str)


def test_generate_checksummed_id_is_verifiable() -> None:
    assert verify_checksummed_id(generate_checksummed_id())


def test_generate_checksummed_id_matches_the_default_shape() -> None:
    assert DEFAULT_PATTERN.match(generate_checksummed_id())


def test_generate_checksummed_id_grouping() -> None:
    # 12 chars payload + 1 check char = 13 chars, grouped by 4 with
    # "-": "XXXX-XXXX-XXXX-X"
    value = generate_checksummed_id(length=12, group_size=4)
    groups = value.split("-")
    assert [len(g) for g in groups] == [4, 4, 4, 1]


def test_generate_checksummed_id_custom_separator() -> None:
    assert generate_checksummed_id(length=5, group_size=3, sep=" ")[3] == " "


def test_generate_checksummed_id_no_separator_length() -> None:
    value = generate_checksummed_id(length=8, group_size=100, sep="")
    assert len(value) == 9


def test_generate_checksummed_id_no_separator_no_separator() -> None:
    value = generate_checksummed_id(length=8, group_size=100, sep="")
    assert "-" not in value


def test_generate_checksummed_id_empty_separator_is_verifiable() -> None:
    assert verify_checksummed_id(generate_checksummed_id(sep=""), sep="")


def test_generate_checksummed_id_is_unique() -> None:
    assert generate_checksummed_id() != generate_checksummed_id()


def test_generate_checksummed_id_length_at_the_maximum() -> None:
    assert verify_checksummed_id(generate_checksummed_id(length=1024))


def test_generate_checksummed_id_fully_specified_literal() -> None:
    with patch(URANDOM, return_value=bytes(range(12))):
        assert generate_checksummed_id() == "0123-4567-89AB-Y"


@pytest.mark.parametrize(
    ("payload", "symbol"), [("10", "*"), ("11", "~"), ("12", "$"), ("13", "="), ("14", "U")]
)
def test_checksum_symbol_extended_symbols(payload: str, symbol: str) -> None:
    assert _checksum_symbol(payload) == symbol


@pytest.mark.parametrize(
    "first_draw", [b"\x01\x00", b"\x01\x01", b"\x01\x02", b"\x01\x03", b"\x01\x04"]
)
def test_generate_checksummed_id_never_emits_an_extended_check_symbol(first_draw: bytes) -> None:
    with patch(URANDOM, side_effect=[first_draw, b"\x00\x01"]):
        assert generate_checksummed_id(length=2, sep="") == "011"


def test_generate_checksummed_id_draws_the_payload_once_more_after_a_rejection() -> None:
    with patch(URANDOM, side_effect=[b"\x01\x00", b"\x00\x01"]) as urandom:
        generate_checksummed_id(length=2)
    assert urandom.call_count == 2


def test_generate_checksummed_id_draws_the_payload_once_when_the_symbol_is_valid() -> None:
    with patch(URANDOM, return_value=b"\x00\x01") as urandom:
        generate_checksummed_id(length=2)
    assert urandom.call_count == 1


def test_generate_checksummed_id_zero_length_raises() -> None:
    with pytest.raises(ValueError, match="length must be greater than 0"):
        generate_checksummed_id(length=0)


def test_generate_checksummed_id_zero_group_size_raises() -> None:
    with pytest.raises(ValueError, match="group_size must be greater than 0"):
        generate_checksummed_id(group_size=0)


def test_generate_checksummed_id_length_above_the_maximum_raises() -> None:
    with pytest.raises(ValueError, match="at most 1024"):
        generate_checksummed_id(length=1025)


@pytest.mark.parametrize("length", [True, 1.5, "8"])
def test_generate_checksummed_id_length_wrong_type_raises(length: object) -> None:
    with pytest.raises(TypeError, match="length must be an int"):
        generate_checksummed_id(length=length)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("group_size", [True, 1.5, "4"])
def test_generate_checksummed_id_group_size_wrong_type_raises(group_size: object) -> None:
    with pytest.raises(TypeError, match="group_size must be an int"):
        generate_checksummed_id(group_size=group_size)  # ty: ignore[invalid-argument-type]


def test_generate_checksummed_id_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        generate_checksummed_id(12)  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize("sep", ["A", "a", "0", "Z", "O", "o", "I", "i", "L", "l"])
def test_generate_checksummed_id_sep_in_the_alphabet_raises(sep: str) -> None:
    with pytest.raises(ValueError, match="sep must not contain a character"):
        generate_checksummed_id(sep=sep)


def test_generate_checksummed_id_sep_multichar_raises() -> None:
    with pytest.raises(ValueError, match="sep must be empty or a single character"):
        generate_checksummed_id(sep="--")


@pytest.mark.parametrize("sep", [1, None, b"-"])
def test_generate_checksummed_id_sep_wrong_type_raises(sep: object) -> None:
    with pytest.raises(TypeError, match="sep must be an instance of"):
        generate_checksummed_id(sep=sep)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("sep", ["*", "~", "$", "=", "U", " ", "_"])
def test_generate_checksummed_id_sep_outside_the_alphabet_is_valid(sep: str) -> None:
    assert verify_checksummed_id(generate_checksummed_id(sep=sep), sep=sep)


###########################################
#     Tests for verify_checksummed_id     #
###########################################


def test_verify_checksummed_id_detects_corruption() -> None:
    value = generate_checksummed_id()
    corrupted = value[:-1] + ("0" if value[-1] != "0" else "1")
    assert not verify_checksummed_id(corrupted)


@pytest.mark.parametrize("index", [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15])
def test_verify_checksummed_id_detects_a_single_substitution(index: int) -> None:
    identifier = "0123-4567-89AB-Y"
    corrupted = identifier[:index] + "Z" + identifier[index + 1 :]
    assert not verify_checksummed_id(corrupted)


@pytest.mark.parametrize("index", [0, 1, 2, 5, 6, 7, 10, 11, 12])
def test_verify_checksummed_id_detects_a_swap_of_adjacent_characters(index: int) -> None:
    identifier = "0123-4567-89AB-Y"
    corrupted = (
        identifier[:index] + identifier[index + 1] + identifier[index] + identifier[index + 2 :]
    )
    assert not verify_checksummed_id(corrupted)


def test_verify_checksummed_id_rejects_garbage() -> None:
    assert not verify_checksummed_id("not-a-valid-id!!")


def test_verify_checksummed_id_rejects_too_short_empty_string() -> None:
    assert not verify_checksummed_id("")


def test_verify_checksummed_id_rejects_too_short_one_character() -> None:
    assert not verify_checksummed_id("A")


def test_verify_checksummed_id_rejects_a_separator_only_identifier() -> None:
    assert not verify_checksummed_id("-")


def test_verify_checksummed_id_accepts_lowercase() -> None:
    assert verify_checksummed_id("0123-4567-89ab-y")


def test_verify_checksummed_id_normalizes_o_as_zero() -> None:
    assert verify_checksummed_id("O123-4567-89AB-Y")


def test_verify_checksummed_id_normalizes_i_as_one() -> None:
    assert verify_checksummed_id("0I23-4567-89AB-Y")


def test_verify_checksummed_id_normalizes_l_as_one() -> None:
    assert verify_checksummed_id("0L23-4567-89AB-Y")


def test_verify_checksummed_id_normalizes_o_in_the_check_symbol() -> None:
    assert verify_checksummed_id("0O")


@pytest.mark.parametrize("identifier", ["10*", "11~", "12$", "13=", "14U", "14u"])
def test_verify_checksummed_id_accepts_the_extended_check_symbols(identifier: str) -> None:
    assert verify_checksummed_id(identifier)


@pytest.mark.parametrize("identifier", [None, 123, b"abc", ["A", "B"]])
def test_verify_checksummed_id_not_a_str(identifier: object) -> None:
    assert not verify_checksummed_id(identifier)  # ty: ignore[invalid-argument-type]


def test_verify_checksummed_id_ascii_look_alike_source_is_valid() -> None:
    payload = "1" * 8
    assert verify_checksummed_id(payload + _checksum_symbol(payload))


@pytest.mark.parametrize(
    "character", ["\N{LATIN SMALL LETTER DOTLESS I}", "\N{FULLWIDTH DIGIT ONE}"]
)
def test_verify_checksummed_id_rejects_non_ascii_look_alikes(character: str) -> None:
    # The dotless i (U+0131) upper-cases to 'I', which normalizes to '1'.
    payload = "1" * 8
    identifier = payload + _checksum_symbol(payload)
    assert not verify_checksummed_id(character + identifier[1:])


def test_verify_checksummed_id_rejects_a_character_outside_the_alphabet() -> None:
    assert not verify_checksummed_id("01!3-4567-89AB-Y")


def test_verify_checksummed_id_rejects_whitespace() -> None:
    assert not verify_checksummed_id(" 0123-4567-89AB-Y")


def test_verify_checksummed_id_custom_separator() -> None:
    assert verify_checksummed_id("0123 4567 89AB Y", sep=" ")


def test_verify_checksummed_id_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        verify_checksummed_id("0123-4567-89AB-Y", "-")  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize("sep", ["A", "o", "--"])
def test_verify_checksummed_id_invalid_sep_raises(sep: str) -> None:
    with pytest.raises(ValueError, match="sep must"):
        verify_checksummed_id("0123-4567-89AB-Y", sep=sep)


def test_verify_checksummed_id_sep_wrong_type_raises() -> None:
    with pytest.raises(TypeError, match="sep must be an instance of"):
        verify_checksummed_id("0123-4567-89AB-Y", sep=1)  # ty: ignore[invalid-argument-type]


######################################
#     Tests for _checksum_symbol     #
######################################


def test_checksum_symbol_invalid_character_raises() -> None:
    with pytest.raises(ValueError, match="payload contains a character outside"):
        _checksum_symbol("!")
