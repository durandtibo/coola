from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import patch

import pytest

from coola.hashing import (
    HasherRegistry,
    PathHasher,
    ResolvedPathHasher,
    get_default_registry,
    tagged,
)
from tests.unit.hashing.helpers import hash_with, leaf_hash


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: PathHasher()})


################################
#     Tests for PathHasher     #
################################


def test_path_hasher_repr() -> None:
    assert repr(PathHasher()) == "PathHasher()"


def test_path_hasher_str() -> None:
    assert str(PathHasher()) == "PathHasher()"


def test_path_hasher_hash_returns_str(tmp_path: Path, registry: HasherRegistry) -> None:
    assert isinstance(hash_with(PathHasher(), tmp_path / "file.txt", registry), str)


def test_path_hasher_hash_default_length(tmp_path: Path, registry: HasherRegistry) -> None:
    assert len(hash_with(PathHasher(), tmp_path / "file.txt", registry)) == 64


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(16, id="16"),
        pytest.param(32, id="32"),
        pytest.param(64, id="64-default"),
        pytest.param(128, id="128"),
    ],
)
def test_path_hasher_hash_length(length: int, tmp_path: Path, registry: HasherRegistry) -> None:
    result = hash_with(PathHasher(), tmp_path / "file.txt", registry, length=length)
    assert len(result) == length


def test_path_hasher_hash_is_deterministic(tmp_path: Path, registry: HasherRegistry) -> None:
    hasher = PathHasher()
    path = tmp_path / "file.txt"
    assert hash_with(hasher, path, registry) == hash_with(hasher, path, registry)


def test_path_hasher_hash_equal_paths_same_hash(tmp_path: Path, registry: HasherRegistry) -> None:
    hasher = PathHasher()
    path_a = tmp_path / "file.txt"
    path_b = tmp_path / "file.txt"
    assert hash_with(hasher, path_a, registry) == hash_with(hasher, path_b, registry)


def test_path_hasher_hash_different_paths_different_hashes(
    tmp_path: Path, registry: HasherRegistry
) -> None:
    hasher = PathHasher()
    path_a = tmp_path / "a.txt"
    path_b = tmp_path / "b.txt"
    assert hash_with(hasher, path_a, registry) != hash_with(hasher, path_b, registry)


def test_path_hasher_hash_ignore_unhashable_has_no_effect(
    tmp_path: Path, registry: HasherRegistry
) -> None:
    # PathHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = PathHasher()
    path = tmp_path / "file.txt"
    assert hash_with(hasher, path, registry) == hash_with(
        hasher, path, registry, ignore_unhashable=True
    )


def test_path_hasher_hash_does_not_use_registry(tmp_path: Path, registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = PathHasher()
    path = tmp_path / "file.txt"
    assert hash_with(hasher, path, registry) == hash_with(hasher, path, empty_registry)


def test_path_hasher_hash_is_the_hash_of_the_leaf(tmp_path: Path, registry: HasherRegistry) -> None:
    hasher = PathHasher()
    path = tmp_path / "file.txt"
    assert hash_with(hasher, path, registry) == leaf_hash(tagged("Path", path.as_posix()))


def test_path_hasher_encode_relative_segments_are_kept() -> None:
    assert PathHasher().encode(Path("a/../b")) == b"Path\x00a/../b"


def test_path_hasher_encode_relative_and_absolute_paths_differ() -> None:
    assert PathHasher().encode(Path("a")) != PathHasher().encode(Path("a").resolve())


def test_path_hasher_encode_pure_path() -> None:
    assert PathHasher().encode(PurePosixPath("a/b")) == b"Path\x00a/b"


def test_path_hasher_encode_windows_path_uses_the_posix_form() -> None:
    assert PathHasher().encode(PureWindowsPath("a\\b")) == b"Path\x00a/b"


def test_path_hasher_encode_is_case_sensitive() -> None:
    assert PathHasher().encode(PurePosixPath("A")) != PathHasher().encode(PurePosixPath("a"))


def test_path_hasher_encode_does_not_touch_the_filesystem() -> None:
    with patch.object(Path, "resolve", side_effect=AssertionError) as resolve:
        PathHasher().encode(Path("a"))
    resolve.assert_not_called()


def test_default_registry_hashes_paths_without_resolving_path() -> None:
    registry = get_default_registry()
    assert registry.hash(Path("a")) == leaf_hash(tagged("Path", "a"))


def test_default_registry_hashes_paths_without_resolving_pure_path() -> None:
    registry = get_default_registry()
    assert registry.hash(PurePosixPath("a")) == leaf_hash(tagged("Path", "a"))


########################################
#     Tests for ResolvedPathHasher     #
########################################


def test_resolved_path_hasher_repr() -> None:
    assert repr(ResolvedPathHasher()) == "ResolvedPathHasher()"


def test_resolved_path_hasher_hash_resolves_path(tmp_path: Path) -> None:
    registry = HasherRegistry()
    hasher = ResolvedPathHasher()
    direct = tmp_path / "file.txt"
    indirect = tmp_path / "sub" / ".." / "file.txt"
    assert hash_with(hasher, direct, registry) == hash_with(hasher, indirect, registry)


def test_resolved_path_hasher_hash_matches_hash_path_resolve(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    result = hash_with(ResolvedPathHasher(), path, HasherRegistry())
    assert result == leaf_hash(tagged("Path", path.resolve().as_posix()))


def test_resolved_path_hasher_encode_resolves_relative_segments(tmp_path: Path) -> None:
    hasher = ResolvedPathHasher()
    assert hasher.encode(tmp_path / "sub" / ".." / "file.txt") == hasher.encode(
        tmp_path / "file.txt"
    )


def test_resolved_path_hasher_encode_resolves_relative_to_absolute() -> None:
    assert ResolvedPathHasher().encode(Path("a")) == tagged("Path", Path("a").resolve().as_posix())


def test_resolved_path_hasher_encode_resolve_failure_uses_the_unresolved_path() -> None:
    with patch.object(Path, "resolve", side_effect=OSError):
        assert ResolvedPathHasher().encode(Path("a/b")) == b"Path\x00a/b"
