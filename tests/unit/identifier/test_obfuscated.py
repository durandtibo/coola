from __future__ import annotations

import pytest

from coola.identifier import decode_obfuscated_id, generate_obfuscated_id
from coola.identifier.obfuscated import _multiplier_and_increment


def test_generate_obfuscated_id_returns_str() -> None:
    assert isinstance(generate_obfuscated_id(1), str)


def test_generate_obfuscated_id_roundtrip_values() -> None:
    for number in (0, 1, 42, 1234, 2**32, 2**64 - 1):
        encoded = generate_obfuscated_id(number, salt="orders")
        assert decode_obfuscated_id(encoded, salt="orders") == number


def test_generate_obfuscated_id_different_salt_differs() -> None:
    assert generate_obfuscated_id(42, salt="a") != generate_obfuscated_id(42, salt="b")


def test_generate_obfuscated_id_decode_wrong_salt_gives_wrong_value() -> None:
    encoded = generate_obfuscated_id(42, salt="right")
    assert decode_obfuscated_id(encoded, salt="wrong") != 42


def test_generate_obfuscated_id_min_length() -> None:
    # The affine map (multiply then add a salt-derived increment) means
    # 0 no longer necessarily encodes to the shortest possible string,
    # so padding can only be asserted as a lower bound, not exact
    # equality: use a min_length comfortably above the base62 length of
    # 2**64 - 1 (11 characters) so padding is actually exercised.
    encoded = generate_obfuscated_id(0, salt="", min_length=16)
    assert len(encoded) == 16
    assert decode_obfuscated_id(encoded, salt="") == 0


def test_generate_obfuscated_id_min_length_shorter_than_natural_length_has_no_effect() -> None:
    encoded = generate_obfuscated_id(2**64 - 1, salt="orders", min_length=1)
    assert decode_obfuscated_id(encoded, salt="orders") == 2**64 - 1


def test_generate_obfuscated_id_zero_is_not_fixed_point() -> None:
    # A purely multiplicative map would send 0 to 0 regardless of salt,
    # revealing which encoded value corresponds to 0. The added,
    # salt-derived increment avoids that.
    assert generate_obfuscated_id(0, salt="orders") != "0"


def test_generate_obfuscated_id_default_salt_still_roundtrips() -> None:
    encoded = generate_obfuscated_id(42)
    assert decode_obfuscated_id(encoded) == 42


def test_generate_obfuscated_id_negative_raises() -> None:
    with pytest.raises(ValueError, match="number must fit in 64 bits"):
        generate_obfuscated_id(-1)


def test_generate_obfuscated_id_too_large_raises() -> None:
    with pytest.raises(ValueError, match="number must fit in 64 bits"):
        generate_obfuscated_id(2**64)


def test_decode_obfuscated_id_invalid_character_raises() -> None:
    with pytest.raises(ValueError, match="encoded contains a character outside"):
        decode_obfuscated_id("not!valid")


def test_generate_obfuscated_id_negative_min_length_raises() -> None:
    with pytest.raises(ValueError, match="min_length must be non-negative"):
        generate_obfuscated_id(1, min_length=-1)


def test_decode_obfuscated_id_empty_raises() -> None:
    with pytest.raises(ValueError, match="encoded must not be empty"):
        decode_obfuscated_id("")


def test_generate_obfuscated_id_single_pair_does_not_reveal_other_ids() -> None:
    # With a purely multiplicative map, one known (plaintext, obfuscated)
    # pair is enough to recover the multiplier (multiplier = obfuscated
    # * modinv(plaintext) mod 2**64) and thus decode every other ID under
    # the same salt. The affine map's extra, additive term cannot be
    # solved for from a single pair, so that one-line attack must no
    # longer recover the id for a different plaintext.
    mask = 2**64 - 1
    known_plaintext, known_salt = 1001, "orders"  # must be odd to be invertible mod 2**64
    known_encoded = generate_obfuscated_id(known_plaintext, salt=known_salt)
    known_obfuscated = _decode_base62_for_test(known_encoded)
    guessed_multiplier = (known_obfuscated * pow(known_plaintext, -1, 1 << 64)) & mask

    other_plaintext = 1002
    other_encoded = generate_obfuscated_id(other_plaintext, salt=known_salt)
    other_obfuscated = _decode_base62_for_test(other_encoded)
    assert ((other_plaintext * guessed_multiplier) & mask) != other_obfuscated


def _decode_base62_for_test(encoded: str) -> int:
    # Mirrors coola.identifier.obfuscated's private base62 decoding, kept
    # separate from the module's own decode_obfuscated_id (which also
    # undoes the affine map) since this test needs the raw obfuscated
    # integer, before subtracting the increment.
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    number = 0
    for char in encoded:
        number = number * len(alphabet) + alphabet.index(char)
    return number


def test_generate_obfuscated_id_encodes_to_zero_for_the_right_number() -> None:
    # _encode_base62's `number == 0` branch (returning "0" directly,
    # rather than falling through the divmod loop) is only reachable
    # when the affine map itself produces exactly 0, i.e. for the one
    # plaintext solving `number * multiplier + increment == 0 (mod
    # 2**64)`. Solve for that plaintext so the branch gets exercised
    # here, instead of relying on hitting it by chance.
    salt = "orders"
    multiplier, increment = _multiplier_and_increment(salt)
    mask = 2**64 - 1
    zero_plaintext = (-increment * pow(multiplier, -1, 1 << 64)) & mask

    encoded = generate_obfuscated_id(zero_plaintext, salt=salt)
    assert encoded == "0"
    assert decode_obfuscated_id(encoded, salt=salt) == zero_plaintext


def test_generate_obfuscated_id_sequential_numbers_differ_by_multiplier_only() -> None:
    # Documented, intentional residual weakness of the affine map: since
    # the additive increment is the same for every call under a given
    # salt, it cancels out in the difference between two obfuscated
    # values, so consecutive plaintexts still differ by exactly the
    # multiplier -- this test pins down that the increment does not
    # (and per the docstring, cannot) change that.
    mask = 2**64 - 1
    a = _decode_base62_for_test(generate_obfuscated_id(500, salt="orders"))
    b = _decode_base62_for_test(generate_obfuscated_id(501, salt="orders"))
    c = _decode_base62_for_test(generate_obfuscated_id(502, salt="orders"))
    assert (b - a) & mask == (c - b) & mask
