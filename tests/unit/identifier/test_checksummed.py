from __future__ import annotations

import pytest

from coola.identifier import generate_checksummed_id, verify_checksummed_id
from coola.identifier.checksummed import _checksum_symbol


def test_generate_checksummed_id_returns_str() -> None:
    assert isinstance(generate_checksummed_id(), str)


def test_generate_checksummed_id_is_verifiable() -> None:
    assert verify_checksummed_id(generate_checksummed_id())


def test_generate_checksummed_id_many_are_verifiable() -> None:
    for _ in range(200):
        assert verify_checksummed_id(generate_checksummed_id())


def test_generate_checksummed_id_grouping() -> None:
    # 12 chars payload + 1 check char = 13 chars, grouped by 4 with
    # "-": "XXXX-XXXX-XXXX-X"
    value = generate_checksummed_id(length=12, group_size=4)
    groups = value.split("-")
    assert [len(g) for g in groups] == [4, 4, 4, 1]


def test_generate_checksummed_id_no_separator() -> None:
    value = generate_checksummed_id(length=8, group_size=100, sep="")
    assert len(value) == 9
    assert "-" not in value


def test_generate_checksummed_id_is_unique() -> None:
    assert generate_checksummed_id() != generate_checksummed_id()


def test_verify_checksummed_id_detects_corruption() -> None:
    value = generate_checksummed_id()
    corrupted = value[:-1] + ("0" if value[-1] != "0" else "1")
    assert not verify_checksummed_id(corrupted)


def test_verify_checksummed_id_rejects_garbage() -> None:
    assert not verify_checksummed_id("not-a-valid-id!!")


def test_verify_checksummed_id_rejects_too_short() -> None:
    assert not verify_checksummed_id("")
    assert not verify_checksummed_id("A")


def test_generate_checksummed_id_zero_length_raises() -> None:
    with pytest.raises(ValueError, match="length must be positive"):
        generate_checksummed_id(length=0)


def test_generate_checksummed_id_zero_group_size_raises() -> None:
    with pytest.raises(ValueError, match="group_size must be positive"):
        generate_checksummed_id(group_size=0)


def test_generate_checksummed_id_sep_overlapping_alphabet_raises() -> None:
    with pytest.raises(ValueError, match="sep must not contain a character"):
        generate_checksummed_id(sep="A")


def test_generate_checksummed_id_sep_overlapping_check_symbol_raises() -> None:
    with pytest.raises(ValueError, match="sep must not contain a character"):
        generate_checksummed_id(sep="*")


def test_checksum_symbol_invalid_character_raises() -> None:
    with pytest.raises(ValueError, match="payload contains a character outside"):
        _checksum_symbol("!")


def test_verify_checksummed_id_accepts_lowercase() -> None:
    assert verify_checksummed_id(generate_checksummed_id().lower())


def test_verify_checksummed_id_normalizes_o_as_zero() -> None:
    payload = "0123456789AB"
    value = payload + _checksum_symbol(payload)
    assert verify_checksummed_id(value.replace("0", "O"), sep="")


def test_verify_checksummed_id_normalizes_i_as_one() -> None:
    payload = "0123456789AB"
    value = payload + _checksum_symbol(payload)
    assert verify_checksummed_id(value.replace("1", "I"), sep="")


def test_verify_checksummed_id_normalizes_l_as_one() -> None:
    payload = "0123456789AB"
    value = payload + _checksum_symbol(payload)
    assert verify_checksummed_id(value.replace("1", "L"), sep="")


def test_generate_checksummed_id_sep_empty_is_verifiable() -> None:
    value = generate_checksummed_id(sep="")
    assert "-" not in value
    assert verify_checksummed_id(value, sep="")


def test_generate_checksummed_id_sep_multichar_raises() -> None:
    with pytest.raises(ValueError, match="sep must be empty or a single character"):
        generate_checksummed_id(sep="--")
