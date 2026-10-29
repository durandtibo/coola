# Architecture of `coola.identifier`

:book: This page describes the design of the `coola.identifier` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../../uguide/identifier.md#properties); the API is in the [reference](../../refs/identifier.md).

## Overview

The package has **two families and one clock rule**. The content-derived identifiers
(`content.py`) are a function of the data through `coola.hashing`; the time-ordered identifiers
(`ulid.py`, `uuid7.py`, `objectid.py`, `snowflake.py`) are a **format** (a bit layout and an encoding)
on top of **one engine**, `TickSequence` (`sequence.py`). A format module has no lock, no clock and no
fork code: it asks the engine for a `(tick, counter)` pair and packs it. The engine has three cases and
none of them raises or waits: a new tick restarts the counter (random, or `0` for Snowflake whose
uniqueness comes from its `worker_id`), the same tick or a clock that went back increments it, and an
exhausted counter borrows the next tick, so the timestamp of an identifier can run ahead of the clock
until the clock catches up. The state is per object and protected by a lock, and
`coola.utils.fork.track_fork` resets it in a forked child; a `SnowflakeIdGenerator`, whose `worker_id`
the child cannot choose, raises there instead. `crockford.py` is the one Crockford Base32 encoder and
decoder (ULID and the checksummed identifiers), and the range and type checks come from
`coola.validation`. The extractors are the strict inverse of the generators: they accept the
canonical form only. See the [user guide](../../uguide/identifier.md#properties) for the properties of
the package.

## Design constraints

1. **Simple and easy to maintain.** The package is 1,861 lines of source (2,659 lines of tests) for
   eleven generators, and four different implementations of "an identifier that increases with the
   time". The target removes code that exists only because of the shape of the current API (the
   Snowflake spin-wait, the persistence arguments, two module-level default generators with the same
   name, a duplicated decoder) instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias,
   no deprecation period, no `generate_snowflake_id` kept "for convenience". The release notes list
   the changes. One exception is deliberate and free: the stable identifiers keep their values
   (golden vectors), because they are persisted by users.
3. **Scope.** A generator stays if it is hard to write correctly (a bit layout, an encoding, a
   monotonic counter, a content hash) **and** safe to use. A one-line wrapper of the standard library
   and a primitive that looks like encryption do not stay.
4. **No slower than before.** The new design is faster than the previous version for the four
   time-ordered generators; the benchmark gate must hold.
5. **Consistent with the siblings.** Strict types and ranges from `coola.validation`, keyword-only
   options, `LazySingleton` where a lazily built process-wide object is needed, `coola.utils.fork`
   for the reset after a fork, golden vectors for the formats that users persist.

## Architecture in detail

```
coola/identifier/
    __init__.py       the public names, the package docstring with the Properties
    content.py        generate_stable_content_id, generate_stable_uuid5        (uses coola.hashing)
    sequence.py       TickSequence: strictly increasing (tick, counter) pairs  (not exported)
    crockford.py      ALPHABET, encode, decode: the one Crockford Base32       (not exported)
    ulid.py           generate_ulid, extract_ulid_timestamp_ms                 (TickSequence 80 bits)
    uuid7.py          generate_uuid7, extract_uuid7_timestamp_ms               (TickSequence 74 bits)
    objectid.py       generate_object_id, extract_object_id_timestamp_s        (TickSequence 64 bits, seconds)
    snowflake.py      SnowflakeIdGenerator, extract_snowflake_timestamp_ms     (TickSequence 12 bits, from 0)
    checksummed.py    generate_checksummed_id, verify_checksummed_id           (crockford)
```

Removed modules: `uuid4.py`, `nanoid.py`, `prefixed.py`, `obfuscated.py`, `uuid5.py` (merged into
`content.py`), `monotonic.py` (renamed `sequence.py`), `validation.py` (split: the range checks go to
`coola.validation`, the alphabet and the decoder to `crockford.py`).

There are two families and nothing else:

- **Content-derived** (`content.py`): a function of the data, through `hashing`.
- **Time-ordered** (`ulid`, `uuid7`, `objectid`, `snowflake`): a **format** (a bit layout and an
  encoding) on top of **one engine** (`TickSequence`). A format module contains no lock, no clock
  and no fork code: it asks the engine for a `(tick, counter)` pair and packs it.

`checksummed.py` is the only module of a third kind (a random code for a human). It stays because the
Crockford checksum is a real piece of work that nobody should rewrite, and it shares the decoder with
ULID.

### The engine: one clock rule

```python
class TickSequence:
    """Strictly increasing ``(tick, counter)`` pairs, from any number of threads."""

    def __init__(self, *, bits: int, tick_ns: int = 1_000_000, random_start: bool = True) -> None:
        self._max = (1 << bits) - 1
        self._bits, self._tick_ns, self._random_start = bits, tick_ns, random_start
        self._lock = threading.Lock()
        self._tick, self._counter = -1, 0
        track_fork(self)

    def next(self) -> tuple[int, int]:
        with self._lock:
            now = time.time_ns() // self._tick_ns
            if now > self._tick:  # a new tick: the counter restarts
                self._tick, self._counter = now, self._start()
            elif self._counter < self._max:  # the same tick, or the clock moved backward
                self._counter += 1
            else:  # the counter is exhausted: borrow the next tick
                self._tick, self._counter = self._tick + 1, self._start()
            return self._tick, self._counter

    def _start(self) -> int:
        return secrets.randbits(self._bits) if self._random_start else 0

    def _reset_after_fork(self) -> None:
        self._lock = threading.Lock()  # a thread that no longer exists may hold it
        self._tick, self._counter = -1, 0
```

This is `MonotonicRandom` (81 lines) with two parameters: the length of a tick (a millisecond, or a
second for ObjectId) and the start of the counter (random for the identifiers that must not collide
across processes, `0` for Snowflake whose uniqueness comes from the `worker_id`). It replaces the
clock logic of `SnowflakeIdGenerator` (about 120 lines, one `RuntimeError`, one `time.sleep`, one
state fix-up) and the counter of `ObjectIdGenerator`.

Design points:

- **The three cases of `next` are the whole rule.** A new tick restarts, the same tick (or a clock
  that went back) increments, an exhausted counter borrows the next tick. No case raises or waits,
  so no case needs a lock to be held for longer than a few operations.
- **Strictly increasing is a property of the pair**, and the pair is packed with the tick in the high
  bits, so the packed value (and its string form for ULID, UUIDv7 and ObjectId) sorts like the pair.
- **The timestamp may run ahead of the wall clock** (burst or backward step); this is stated in the
  guide. For Snowflake it happens above 4,096 identifiers per
  millisecond (4 million per second).
- **The validation of the packed timestamp stays in the format module** (a ULID timestamp is 48 bits,
  a Snowflake one is 41 bits from its epoch), after `next`, so a borrowed tick that leaves the range
  raises a `ValueError` like an explicit one.

### The formats

| Module         | Layout                                                                | Generator                                                           | Extractor                                                                                         |
| -------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `ulid.py`      | 48-bit ms, 80 random bits, 26 characters Crockford Base32             | `generate_ulid(*, timestamp_ms=None)`                               | `extract_ulid_timestamp_ms(ulid)`: 26 characters, `crockford.decode`, first character at most `7` |
| `uuid7.py`     | 48-bit ms, version `7`, 74 random bits, variant `10` (RFC 9562)       | `generate_uuid7(*, timestamp_ms=None)`                              | `extract_uuid7_timestamp_ms(uuid7)`: canonical 36-character form, version 7, variant RFC 9562     |
| `objectid.py`  | 32-bit seconds, 64 random bits, 24 lowercase hex characters           | `generate_object_id(*, timestamp_s=None)`                           | `extract_object_id_timestamp_s(object_id)`: `[0-9a-f]{24}`                                        |
| `snowflake.py` | 41-bit ms from 2024-01-01, 10-bit `worker_id`, 12-bit sequence, `int` | `SnowflakeIdGenerator(*, worker_id).generate(*, timestamp_ms=None)` | `extract_snowflake_timestamp_ms(snowflake_id)`: `int` in `[0, 2**63)`                             |

ULID, UUIDv7 and ObjectId are the same function: pack `(tick << width) | counter`, then encode. A
`generate_ulid` is about ten lines:

```python
_SEQUENCE = TickSequence(bits=80)


def generate_ulid(*, timestamp_ms: int | None = None) -> str:
    if timestamp_ms is None:
        timestamp_ms, counter = _SEQUENCE.next()
    else:
        counter = secrets.randbits(80)
    validate_timestamp_ms(timestamp_ms)  # validate_int, then validate_closed_range(0, 2**48 - 1)
    return crockford.encode((timestamp_ms << 80) | counter, length=26)
```

Details that change:

- **ObjectId** is stored as 4 bytes of seconds followed by 64 bits drawn at the start of each second
  and incremented, instead of "5-byte per-instance value plus 3-byte counter". The 24 hexadecimal
  characters, the timestamp field and the sort order are the same; the identifier is now strictly
  increasing within a process, needs no per-process value to reset after a fork, and
  `ObjectIdGenerator` and `get_default_generator` disappear. Only a tool that decodes the process
  field notices.
- **Snowflake** takes its `worker_id` in the constructor (required: no default `0`), owns one
  `TickSequence(bits=12, random_start=False)`, and has no module-level function and no default
  singleton: a process-wide generator has one `worker_id`, which only the caller knows. In a forked
  child, `_reset_after_fork` sets a flag and `generate` raises
  `RuntimeError("SnowflakeIdGenerator cannot be used in a forked process: create one with its own worker_id")`.
  A silent duplicate (101 of 200 forks before, ) becomes a loud error. `last_timestamp_ms`,
  `sequence`, `_SPIN_SLEEP_S` and the `RuntimeError` for a backward clock are deleted.
- **Crockford** has one decoder: case-insensitive, `O` is `0`, `I` and `L` are `1` (the rule of the
  specification, now also applied to ULID), and `encode(value, length)` is the loop of the previous
  `_encode_base32` generalised. `verify_checksummed_id` and `extract_ulid_timestamp_ms` call it.
- **Validation** is `validate_int` then `validate_closed_range` of `coola.validation`, with the
  parameter name, written inline (two lines per parameter). The 48-bit timestamp check of ULID and
  UUIDv7 is repeated in the two modules instead of being a shared private helper: it is two lines,
  and `CLAUDE.md` forbids private helpers imported across modules.

### The content-derived identifiers

```python
def generate_stable_content_id(
    data: object,
    *,
    registry: HasherRegistry | None = None,
    length: int = 64,
    ignore_unhashable: bool = False,
) -> str:
    if registry is None:
        registry = get_stable_registry()
    return hash_object(data, registry=registry, length=length, ignore_unhashable=ignore_unhashable)


def generate_stable_uuid5(
    data: object,
    *,
    registry: HasherRegistry | None = None,
    namespace: uuid.UUID = NAMESPACE,
    ignore_unhashable: bool = False,
) -> str:
    digest = generate_stable_content_id(
        data, registry=registry, length=128, ignore_unhashable=ignore_unhashable
    )
    return str(uuid.uuid5(namespace, digest))
```

Same computation as before, so **the values do not change** (golden vectors). One module,
one docstring for the shared parameters, and a corrected note: the UUID is limited by its 122 random
bits (birthday bound 2^61), not by SHA-1; use `generate_stable_content_id` when more bits are needed.

### The checksummed identifiers

- `generate_checksummed_id(*, length=12, group_size=4, sep="-")` and
  `verify_checksummed_id(identifier, *, sep="-")`: keyword-only options.
- **The generator never emits the extended check symbols.** If the mod-37 remainder of the payload is
  32 or more, the payload is drawn again (5 of 37 payloads, 1.16 draws on average), so the identifier
  is made of `0-9A-Z` and the separator: safe in a shell, a URL and a spreadsheet. The cost is
  `log2(37/32)`, that is 0.21 bit of the entropy of a 60-bit payload. `verify_checksummed_id` still
  accepts the five symbols of the specification, so identifiers from another implementation verify.
- `sep` must not be in the 32-character alphabet (it is stripped before the check), which is shorter
  to state than "the extended alphabet".
- `verify_checksummed_id` returns `False` and never raises, for a wrong type, a non-ASCII character
  (the dotless `i` upper-cases to `I`) or a wrong alphabet; the rule stays the one of the previous version.

## Public API

```python
from coola.identifier import (
    generate_stable_content_id,  # (data, *, registry=None, length=64, ignore_unhashable=False)
    generate_stable_uuid5,  # (data, *, registry=None, namespace=..., ignore_unhashable=False)
    generate_ulid,  # (*, timestamp_ms=None)
    extract_ulid_timestamp_ms,  # (ulid)
    generate_uuid7,  # (*, timestamp_ms=None)
    extract_uuid7_timestamp_ms,  # (uuid7)
    generate_object_id,  # (*, timestamp_s=None)
    extract_object_id_timestamp_s,  # (object_id)
    SnowflakeIdGenerator,  # (*, worker_id)   .generate(*, timestamp_ms=None) -> int
    extract_snowflake_timestamp_ms,  # (snowflake_id)
    generate_checksummed_id,  # (*, length=12, group_size=4, sep="-")
    verify_checksummed_id,  # (identifier, *, sep="-")
)
```

Removed: `generate_uuid4`, `generate_nano_id`, `generate_prefixed_id`, `generate_obfuscated_id`,
`decode_obfuscated_id`, `generate_snowflake_id`, `ObjectIdGenerator`, `MonotonicRandom`, the two
`get_default_generator` and `identifier.validation`. Renamed: `extract_object_id_timestamp` to
`extract_object_id_timestamp_s` (the unit in the name, like `_ms`). Moved to keyword-only:
`timestamp_ms`, `timestamp`, `worker_id`, `length`, `group_size`, `sep`.

What to use instead of the removed names (the migration page and the guide say it):

| Removed                  | Use                                                                            |
| ------------------------ | ------------------------------------------------------------------------------ |
| `generate_uuid4()`       | `str(uuid.uuid4())`                                                            |
| `generate_nano_id(n, a)` | `secrets.token_urlsafe(16)`, or `"".join(secrets.choice(a) for _ in range(n))` |
| `generate_prefixed_id`   | `f"cus_{generate_ulid()}"`                                                     |
| `generate_obfuscated_id` | a library made for it (`sqids`), or a keyed permutation; never an affine map   |
| `generate_snowflake_id`  | `SnowflakeIdGenerator(worker_id=...)` created once per process                 |
| `ObjectIdGenerator`      | `generate_object_id()`; pass `timestamp_s` for a deterministic test            |

## Dependents

| Where                                                             | Change                                                                                                                                 |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/hashing/test_golden.py`, `tests/integration/hashing/` | None: they use the stable identifiers, which keep their values. `test_content.py` gets its own golden vectors for the two identifiers. |

## Decision record

All eight decisions were taken by the maintainers and follow the recommendation. The alternatives are
kept to explain what was rejected.

| Decision | Taken                                                                                      |
| -------- | ------------------------------------------------------------------------------------------ |
| D1       | Option A: remove `uuid4`, `nanoid`, `prefixed` and `obfuscated`; keep the other generators |
| D2       | Keep the checksummed identifier, with rejection sampling                                   |
| D3       | `SnowflakeIdGenerator(*, worker_id)`, no module-level function                             |
| D4       | The clock rule never raises or blocks                                                      |
| D5       | ObjectId: 4 bytes of seconds and a 64-bit incremented random counter                       |
| D6       | Strict extractors: only the canonical form                                                 |
| D7       | Keep our own `generate_uuid7` until the minimum Python version is 3.14                     |
| D8       | The stable identifiers stay in `coola.identifier`; contract forbids `hashing` importing it |

**D1. How much of the package stays?** _Recommended: keep the two stable identifiers, the four
time-ordered formats and the checksummed identifier; remove `uuid4`, `nanoid`, `prefixed` and
`obfuscated`._ The rule is the one of: a generator stays if it is hard to write correctly
and safe. `uuid4`, `prefixed` and `nanoid` are a line of the standard library, cost 3
modules (4 with `obfuscated`), and each one has to be documented, tested and kept consistent with the others.
`obfuscated` is the one with a real argument against it: it looks like encryption, leaks the parity,
is broken by two pairs, and needs a 47-line module docstring on its limits; a user who needs an opaque public identifier
should get one from a library made for it. Alternatives: (B) keep everything and apply only steps
1, 3, 4, 5, 6 and 7, with `Callable`-based `prefixed` and the fixes of and for the four modules
(about 420 more lines of source and 480 of tests to maintain), or (C) move the whole package to a
separate distribution and keep only `content.py` in `coola`. (C) is the cleanest layering, and the
most work for the users; it is worth considering if the time-ordered formats keep growing.

**Decided: A.**

**D2. Keep the checksummed identifier?** _Recommended: yes, with rejection sampling._ It is the one
module that shares code with ULID (the decoder), its detection claim is true and tested, and the fix
of makes its output usable for what it is for. If D1(C) is chosen it moves with the others.

**Decided: as recommended.**

**D3. Snowflake without a module-level function.** _Recommended: yes._ A process-wide generator has
one `worker_id`, which the library cannot know; before the id is a per-call argument that shares one
sequence between workers and defaults to a value that duplicates identifiers across processes. The
price is one line more in the caller (`SnowflakeIdGenerator(worker_id=3)` once per process), the
benefit is that the wrong code does not exist. Alternative: keep `generate_snowflake_id(worker_id)`
backed by a dictionary of generators keyed by `worker_id`; it keeps the two-line call and adds a
global.

**Decided: yes.**

**D4. The clock rule never raises.** _Recommended: yes, for the four formats._ A burst or a backward
step costs accuracy of the embedded timestamp (it runs ahead), never order or uniqueness, and the
Snowflake `RuntimeError` / `time.sleep` code, its tests and the guide paragraph on NTP disappear.
Alternative: keep the `RuntimeError` for Snowflake only, as a strict clock check; it needs the second
rule back in `snowflake.py`, and the user must retry.

**Decided: as recommended.**

**D5. ObjectId layout.** _Recommended: 4 bytes of seconds and 64 bits of an incremented random
counter._ It is strictly increasing within a process, fork-safe through the engine and 60 lines
shorter than before. The documented MongoDB layout (5-byte per-process value and 3-byte counter) is
lost; it only matters to a tool that decodes those two fields, and `extract_object_id_timestamp_s`
reads the same bytes. Alternative: keep the layout with a per-process value stored in a
`LazySingleton`-built object reset after a fork (the class that exists today).

**Decided: as recommended.**

**D6. Strict extractors.** _Recommended: only the canonical form._ `extract_uuid7_timestamp_ms`
takes the 36-character form that `generate_uuid7` returns (any case), not `{...}` or `urn:uuid:`; the
ObjectId extractor takes 24 hex characters; the ULID extractor takes 26 Crockford characters (with
the `O`/`I`/`L` rule). A caller who has a `uuid.UUID` passes `str(...)`. Alternative: accept
`uuid.UUID | str`; one `isinstance` more, and the same function for the two inputs.

**Decided: as recommended.**

**D7. `generate_uuid7` and Python 3.14.** `uuid.uuid7()` exists in Python 3.14 (checked on 3.14.8)
and `requires-python` is `>=3.11`. _Recommended: keep our own._ When the minimum version reaches
3.14, replace the body of `generate_uuid7` by `str(uuid.uuid7())` after checking that the standard
library keeps the same monotonic guarantee within a process; the extractor stays.

**Decided: as recommended.**

**D8. Layering of the stable identifiers.** _Recommended: keep them in `coola.identifier`, document
that `identifier` depends on `hashing`, and forbid the reverse import by contract._ Moving them into
`coola.hashing` would make `identifier` standalone, but the name users know, and the guide, would
move for little benefit.

**Decided: as recommended.**

## Not covered

- **Unpredictable tokens** (sessions, API keys, password reset links): use `secrets.token_urlsafe`;
  none of these identifiers is a capability.
- **Other formats**: UUIDv6 and v8, KSUID, TypeID, Sqids. The engine makes a new time-ordered format
  a 20-line module; nothing here adds one.
- **Datetime helpers**: `extract_*` return an `int` (milliseconds or seconds since the Unix epoch);
  a `datetime` is one call away and keeping the return type small avoids a time-zone decision.
- **A pluggable clock.** The tests use the explicit timestamp and a patched `time.time_ns` for the
  engine only; a `clock` parameter on every generator is not worth its surface.
- **Allocation of Snowflake worker ids** across hosts (a database row, an environment variable, a
  coordinator): the caller's job, stated in the guide.
- **The format version of the stable identifiers.** A `HASH_FORMAT_VERSION` mixed into the UUID5
  belongs to `hashing`.
- **Windows.** `os.register_at_fork` does not exist there; the fork rules do not apply and the
  `os.fork` tests are skipped, as before.
