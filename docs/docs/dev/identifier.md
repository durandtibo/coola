# Architecture of `coola.identifier`

:book: This page describes the design of the `coola.identifier` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/identifier.md#properties); the API is in the [reference](../refs/identifier.md).

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
canonical form only. See the [user guide](../uguide/identifier.md#properties) for the properties of
the package.

## Design constraints

1. **Simple and easy to maintain.** Eleven generators, one engine for "an identifier that increases
   with the time", and no code that exists only because of the shape of the API (no spin-wait, no
   persistence arguments, no duplicated decoder).
2. **Persisted formats are stable.** No compatibility alias is kept. The stable identifiers keep their
   values (golden vectors), because users persist them.
3. **Scope.** A generator stays if it is hard to write correctly (a bit layout, an encoding, a
   monotonic counter, a content hash) **and** safe to use. A one-line wrapper of the standard library
   and a primitive that looks like encryption do not belong here.
4. **Fast.** The time-ordered generators go through one lock and a few integer operations; the
   benchmark gate must hold.
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

The engine has two parameters: the length of a tick (a millisecond, or a second for ObjectId) and the
start of the counter (random for the identifiers that must not collide across processes, `0` for
Snowflake whose uniqueness comes from the `worker_id`).

Design points:

- **The three cases of `next` are the whole rule.** A new tick restarts, the same tick (or a clock
  that went back) increments, an exhausted counter borrows the next tick. No case raises or waits,
  so no case needs a lock to be held for longer than a few operations.
- **Strictly increasing is a property of the pair**, and the pair is packed with the tick in the high
  bits, so the packed value (and its string form for ULID, UUIDv7 and ObjectId) sorts like the pair.
- **The timestamp may run ahead of the wall clock** (burst or backward step); this is stated in the
  guide. For Snowflake it happens above 4,096 identifiers per millisecond (4 million per second).
  Not raising or sleeping on a backward clock costs the accuracy of the embedded timestamp, never the
  order or the uniqueness; raising for Snowflake only would need a second clock rule in
  `snowflake.py`, and the caller would have to retry.
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

Details of the formats:

- **ObjectId** is 4 bytes of seconds followed by 64 bits drawn at the start of each second and
  incremented. The 24 hexadecimal characters, the timestamp field and the sort order are those of an
  ObjectId; the identifier is strictly increasing within a process and needs no per-process value to
  reset after a fork. The documented MongoDB layout (a 5-byte per-process value and a 3-byte counter)
  is not used, which only matters to a tool that decodes those two fields; `extract_object_id_timestamp_s`
  reads the same bytes.
- **Snowflake** takes its `worker_id` in the constructor (required: no default `0`), owns one
  `TickSequence(bits=12, random_start=False)`, and has no module-level function and no default
  singleton: a process-wide generator has one `worker_id`, which only the caller knows. In a forked
  child, `_reset_after_fork` sets a flag and `generate` raises
  `RuntimeError("SnowflakeIdGenerator cannot be used in a forked process: create one with its own worker_id")`,
  so a silent duplicate becomes a loud error. The price is one more line in the caller
  (`SnowflakeIdGenerator(worker_id=3)` once per process); a `generate_snowflake_id(worker_id)` backed
  by a dictionary of generators would keep the two-line call at the cost of a global.
- **Crockford** has one decoder: case-insensitive, `O` is `0`, `I` and `L` are `1` (the rule of the
  specification, applied to ULID too), and `encode(value, length)`. `verify_checksummed_id` and
  `extract_ulid_timestamp_ms` call it.
- **Validation** is `validate_int` then `validate_closed_range` of `coola.validation`, with the
  parameter name, written inline (two lines per parameter). The 48-bit timestamp check of ULID and
  UUIDv7 is repeated in the two modules instead of being a shared private helper: it is two lines, and
  a private helper must not be imported across modules.
- **Strict extractors** accept the canonical form only. `extract_uuid7_timestamp_ms` takes the
  36-character form that `generate_uuid7` returns (any case), not `{...}` or `urn:uuid:`; the ObjectId
  extractor takes 24 hex characters; the ULID extractor takes 26 Crockford characters. A caller who
  has a `uuid.UUID` passes `str(...)`. Accepting `uuid.UUID | str` would add one `isinstance`.
- **`generate_uuid7` is our own.** `uuid.uuid7()` exists in Python 3.14 and `requires-python` is
  `>=3.11`. When the minimum version reaches 3.14, the body can become `str(uuid.uuid7())` after
  checking that the standard library keeps the same monotonic guarantee within a process; the
  extractor stays.

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

The values are covered by golden vectors. The UUID is limited by its 122 random bits (birthday bound
2^61), not by SHA-1; use `generate_stable_content_id` when more bits are needed. The stable identifiers
live in `coola.identifier`: `identifier` depends on `hashing`, and a contract forbids the reverse
import. Moving them into `coola.hashing` would make `identifier` standalone, but the name users know,
and the guide, would move for little benefit.

### The checksummed identifiers

- `generate_checksummed_id(*, length=12, group_size=4, sep="-")` and
  `verify_checksummed_id(identifier, *, sep="-")`: keyword-only options.
- **The generator never emits the extended check symbols.** If the mod-37 remainder of the payload is
  32 or more, the payload is drawn again (5 of 37 payloads, 1.16 draws on average), so the identifier
  is made of `0-9A-Z` and the separator: safe in a shell, a URL and a spreadsheet. The cost is
  `log2(37/32)`, that is 0.21 bit of the entropy of a 60-bit payload. `verify_checksummed_id` still
  accepts the five symbols of the specification, so identifiers from another implementation verify.
  The checksummed identifier stays in the package because the Crockford checksum is a real piece of
  work that nobody should rewrite, and it shares the decoder with ULID.
- `sep` must not be in the 32-character alphabet (it is stripped before the check), which is shorter
  to state than "the extended alphabet".
- `verify_checksummed_id` returns `False` and never raises, for a wrong type, a non-ASCII character
  (the dotless `i` upper-cases to `I`) or a wrong alphabet.

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

Options are keyword-only (`timestamp_ms`, `timestamp_s`, `worker_id`, `length`, `group_size`, `sep`).

The package deliberately has no generator for a one-line wrapper of the standard library or for a
primitive that looks like encryption. What to use instead:

| Need                         | Use                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------ |
| a random UUID4               | `str(uuid.uuid4())`                                                            |
| a random URL-safe string     | `secrets.token_urlsafe(16)`, or `"".join(secrets.choice(a) for _ in range(n))` |
| a prefixed identifier        | `f"cus_{generate_ulid()}"`                                                     |
| an opaque public identifier  | a library made for it (`sqids`), or a keyed permutation; never an affine map   |
| a Snowflake identifier       | `SnowflakeIdGenerator(worker_id=...)` created once per process                 |
| a deterministic ObjectId     | `generate_object_id(timestamp_s=...)`                                          |

An obfuscated identifier (an affine permutation of an integer) looks like encryption, leaks the parity
and is broken by two pairs, so it needs a long list of caveats; a user who needs an opaque public
identifier should get one from a library made for it.

## Dependents

| Where                                                             | Use                                                                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `tests/unit/hashing/test_golden.py`, `tests/integration/hashing/` | They use the stable identifiers, which keep their values. `test_content.py` has golden vectors for the two identifiers. |

## Design choices and trade-offs

| Choice                                                                                      | Reason                                                                   |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Keep the stable identifiers, the four time-ordered formats and the checksummed identifier   | hard to write correctly and safe                                         |
| The clock rule never raises or blocks                                                       | a burst or a backward step costs accuracy, never order or uniqueness     |
| ObjectId: 4 bytes of seconds and a 64-bit incremented random counter                        | strictly increasing within a process, fork-safe through the engine       |
| `SnowflakeIdGenerator(*, worker_id)`, no module-level function                              | the wrong code (a shared `worker_id`) does not exist                     |
| Checksummed identifier with rejection sampling                                              | the output is usable in a shell, a URL and a spreadsheet                 |

**How much of the package exists.** The rule is "a generator stays if it is hard to write correctly and
safe". A UUID4, a prefixed identifier or a nano ID is a line of the standard library and would cost a
module, a docstring and tests each. The alternatives are to keep everything (more source and tests to
maintain, and a `Callable`-based prefixed generator), or to move the whole package into a separate
distribution and keep only `content.py` in `coola`; the second is the cleanest layering and more work
for users, and is worth considering if the time-ordered formats keep growing.

**The checksummed identifier.** It shares code with ULID (the decoder), its detection claim is true and
tested, and rejection sampling makes its output usable for what it is for. If the package were split,
it would move with the other generators.

**The clock rule.** Raising a `RuntimeError` on a backward clock for Snowflake only would need the second
rule back in `snowflake.py`, a `time.sleep` for the spin-wait, and a guide paragraph on NTP.

**ObjectId.** Keeping the MongoDB layout would need a per-process value stored in a
`LazySingleton`-built object that is reset after a fork.

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
- **The format version of the stable identifiers.** The version is the BLAKE2b `person` parameter of
  `hashing`.
- **Windows.** `os.register_at_fork` does not exist there; the fork rules do not apply and the
  `os.fork` tests are skipped.
