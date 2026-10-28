# `coola.identifier`: target state and the changes to reach it

This document describes the design that `coola.identifier` should have, then the changes to make to
the current implementation (`main`, `11a0a8c0`) to get there. The design follows the rules of the
user guide (`docs/docs/uguide/identifier.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: package layering, lazy imports, strict contracts, forks), and it
stays aligned with the sibling packages `hashing` (the stable identifiers are built on it), `random`
and `validation`. Its aim is a small package that is easy to maintain: **one job per generator, one
clock rule for every time-ordered identifier, one decoder for Crockford Base32, strict inputs and
strict inverses**. Breaking changes are accepted, and the future state is more important than the
past one. The decisions D1 to D8 of section 10 are settled (all follow the recommendation), so the
design below is the one to implement.

Evidence comes from probes and a throw-away prototype run on `main` (Python 3.14.8, scripts kept
outside the repository). Items marked **[verified]** were reproduced; **[reading]** comes from the
code or from an argument. Nothing in `src`, `tests` or `docs` was changed.

## 1. Properties of the sub-package

These are the rules of the user guide, completed with the rules that the code already follows
without saying it, and the ones that are missing or false today. Every design choice below exists to
make one of them true **by construction** (in the shared engine, in the shared decoder, in the
validation) instead of by the discipline of each generator.

| Property                 | Statement                                                                                                                                                                                                                                                    | Enforced by                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| Declared shape           | Every generator returns a `str` of a documented shape (length, alphabet), except `SnowflakeIdGenerator.generate`, which returns an `int` of at most 63 bits. The shape is the same for every call and every input.                                           | one format test per generator, property tests                |
| Random bits are CSPRNG   | Every random bit comes from `secrets` or `os.urandom`, never from `random`. Two identifiers collide with the probability of the birthday bound of their random width (ULID 80 bits, UUIDv7 74, ObjectId 64, UUID5 122).                                      | the generators                                               |
| Identifier, not a secret | An identifier is not an access token: the time-ordered ones are guessable (the next value is the previous one plus one), and the content-derived ones are reproducible. Authorise every access explicitly.                                                   | documented                                                   |
| Content-derived          | `generate_stable_content_id` and `generate_stable_uuid5` give the same value for data that `hash_object` hashes alike, whatever the insertion order of the mappings, the process, the machine and the installed plugins.                                     | `get_stable_registry()`, golden vectors                      |
| Time-ordered             | `generate_ulid`, `generate_uuid7`, `SnowflakeIdGenerator.generate` and `generate_object_id` called with the current time return **strictly increasing** values within a process, from any number of threads: the sort order is the generation order.         | `TickSequence` (one rule)                                    |
| Never raises or blocks   | A time-ordered generator does not raise, sleep or spin when the clock steps backward, when more than one identifier is requested in the same tick or when the counter is exhausted: the timestamp is kept (or advanced by one tick) and the counter goes on. | `TickSequence`                                               |
| Explicit timestamp       | The keyword-only timestamp (`timestamp_ms`, `timestamp_s`) is for tests and backfills. It does not touch the state of the generator and the counter part is random: such identifiers are neither ordered nor guaranteed to be unique.                        | the generators                                               |
| Strict inverse           | `extract_*(generate_*(timestamp=t))` is `t`, and an extractor rejects every string the generator cannot produce (wrong length, wrong alphabet, wrong version, whitespace, wrong type). Case is the only leniency, as the specifications say.                 | one decoder per format                                       |
| Check symbol             | `verify_checksummed_id` is `True` for every identifier of `generate_checksummed_id`, detects any single wrong character and any swap of two adjacent characters, and never raises (`False` for a wrong type or alphabet).                                    | mod-37 Crockford checksum                                    |
| Thread-safe              | Every public callable can be called from several threads without a lock of the caller. The state is per generator, and the only module-level state is the sequence of each default generator.                                                                | one lock per `TickSequence`                                  |
| Fork-safe                | A forked child never continues the sequence of its parent: the sequences and their locks are reset in the child. A `SnowflakeIdGenerator` (whose uniqueness comes from its `worker_id`, which the child cannot choose) raises in the child.                  | `track_fork`, `TickSequence`, the Snowflake flag             |
| Strict inputs            | A parameter of the wrong type raises `TypeError` (`bool` and `float` are not accepted as an `int`), a value out of range raises `ValueError`, both name the parameter and both are raised before the state of any generator changes.                         | `coola.validation` (`validate_int`, `validate_closed_range`) |
| Keyword-only options     | Every option after the data or the identifier is keyword-only (the rule of `hash_object` and of `CLAUDE.md`).                                                                                                                                                | the signatures                                               |
| Lightweight              | The package imports the standard library and `coola.hashing` (for the two stable identifiers), no optional library. A call costs less than 3 µs for the time-ordered generators.                                                                             | lazy imports of `hashing`, the prototype benchmark           |
| Consistent with siblings | The stable identifiers inherit the properties of `hashing` (they can change in a major release of `coola`); the error messages and the validation come from `coola.validation`; the fork reset is the one of `coola.utils.fork`.                             | delegation to `hashing`, `validation`, `utils.fork`          |

Not guaranteed, and documented as such:

- **Order across processes.** Two processes (or two generators) that mint an identifier in the same
  tick (a millisecond, a second for ObjectId) give identifiers that are ordered by the random part,
  so not by the time of the call.
- **A timestamp equal to the wall clock.** After a burst (more than 4,096 Snowflake identifiers in
  one millisecond) or a backward step of the clock, the timestamp of the identifiers runs ahead of
  the clock until the clock catches up. Order and uniqueness win over accuracy.
- **Uniqueness across processes of Snowflake identifiers** when two processes use the same
  `worker_id`: assigning worker numbers is the job of the caller.
- **Stability of the stable identifiers across major versions of `coola`** (they follow `hashing`),
  and across registries other than the stable one.
- **Unguessability** of any identifier (see "Identifier, not a secret").
- **Collision resistance beyond the random width.** A UUID has 122 random bits: its birthday bound is
  2^61 whatever the hash behind it.

## 2. Constraints

1. **Simple and easy to maintain.** The package is 1,861 lines of source (2,659 lines of tests) for
   eleven generators, and four different implementations of "an identifier that increases with the
   time". The target removes code that exists only because of the shape of the current API (the
   Snowflake spin-wait, the persistence arguments, two module-level default generators with the same
   name, a duplicated decoder) instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias,
   no deprecation period, no `generate_snowflake_id` kept "for convenience". The release notes list
   the changes. One exception is deliberate and free: the stable identifiers keep their values
   (golden vectors, section 9), because they are persisted by users.
3. **Scope.** A generator stays if it is hard to write correctly (a bit layout, an encoding, a
   monotonic counter, a content hash) **and** safe to use. A one-line wrapper of the standard library
   and a primitive that looks like encryption do not stay (decision D1).
4. **No slower than today.** The prototype of section 7 is faster than `main` for the four
   time-ordered generators; the benchmark gate of section 9 must hold.
5. **Consistent with the siblings.** Strict types and ranges from `coola.validation`, keyword-only
   options, `LazySingleton` where a lazily built process-wide object is needed, `coola.utils.fork`
   for the reset after a fork, golden vectors for the formats that users persist.

## 3. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Status         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The scope is wider than the library.** Eleven generators and three helpers (1,861 lines, 6 % of `src`), and only the two stable identifiers use the rest of `coola` (`hashing`). No other package of `src` imports `coola.identifier`; the only users in the repository are two `hashing` tests. Four generators are not worth owning: `generate_uuid4` is `str(uuid.uuid4())`, `generate_prefixed_id` is `f"{prefix}_{generate_ulid()}"` plus 60 lines of validation, `generate_nano_id` re-implements `secrets.choice`, and `generate_obfuscated_id` is an affine map that the module itself documents as breakable with two known pairs (a 47-line module docstring on its limits).                                                                | **[reading]**  |
| F2  | **Four implementations of "increases with the time".** ULID and UUIDv7 share `MonotonicRandom` (module-level state, increment, borrow the next millisecond). `SnowflakeIdGenerator` has its own lock, its own clock rule (120 lines: raise `RuntimeError` on a backward clock, spin with `time.sleep` while holding the lock on exhaustion, put the sequence back on a second backward jump). `ObjectIdGenerator` has a third rule (the counter wraps silently, no ordering within a second). The guide therefore has to state three failure modes for what is one property.                                                                                                                                                                            | **[reading]**  |
| F3  | **Snowflake: `worker_id` is a per-call argument, and a fork duplicates identifiers.** The sequence is shared by all the worker ids of an instance, the default `0` is the footgun the docstring warns about, and the state of the generator is kept after a fork on purpose. With the default arguments, a parent and a forked child mint **the same identifier in 101 of 200 trials** (the child continues the sequence of its parent). `last_timestamp_ms` and `sequence` of the constructor exist for the tests and for a persistence that nobody described.                                                                                                                                                                                         | **[verified]** |
| F4  | **Options are positional.** `generate_ulid(timestamp_ms)`, `generate_uuid7(timestamp_ms)`, `generate_snowflake_id(worker_id, timestamp_ms)`, `generate_object_id(timestamp)`, `generate_nano_id(length, alphabet)`, `generate_checksummed_id(length, group_size, sep)`, `verify_checksummed_id(identifier, sep)`, `generate_obfuscated_id(number, salt, min_length)`, `generate_prefixed_id(prefix, generator)`. The unit of the timestamp is in the name for three of them (`_ms`) and not for `generate_object_id(timestamp)` (seconds).                                                                                                                                                                                                              | **[reading]**  |
| F5  | **Weak input validation.** `bool` is accepted as a timestamp, a `worker_id` and a length (`generate_ulid(True)` returns a ULID of 1 ms, `generate_snowflake_id(worker_id=True)` works). A `float` raises whatever the code touches first: `AttributeError: 'float' object has no attribute 'to_bytes'` (ULID, ObjectId), `TypeError: unsupported operand type(s) for <<` (UUIDv7, Snowflake). `coola.validation.validate_int` and `validate_closed_range` exist, and `identifier/validation.py` re-implements the range check with another message.                                                                                                                                                                                                     | **[verified]** |
| F6  | **The extractors are not strict inverses.** `extract_object_id_timestamp("65 92 0a 80 aa bb cc dd ee ff 00 11"[:24])` returns `1704069760` instead of raising (`bytes.fromhex` skips whitespace; `bytes` and upper case are accepted too). `extract_uuid7_timestamp_ms` accepts `{...}`, `urn:uuid:...` and the form without hyphens, reports a wrong variant as "got version None", and raises `AttributeError` for a `uuid.UUID`. `extract_ulid_timestamp_ms` rejects `O`, `I` and `L` that `verify_checksummed_id` accepts: there are two Crockford decoders with two rules. `extract_snowflake_timestamp_ms` says "64 bits" and checks 63. A non-`str` raises from `len()`.                                                                         | **[verified]** |
| F7  | **Stale and wrong documentation.** The package docstring says "identifiers for nested data" (two of eleven are). `ObjectIdGenerator` documents a "per-process" value that is per instance. The guide's **Properties** table and "Which one should I use?" omit ObjectId, NanoID, the checksummed and obfuscated identifiers; the guide says "The four time-ordered generators" and lists three in the property. The architecture guide says `identifier` is a "standalone helper" and there is no import-linter contract, although the stable identifiers import `coola.hashing`. `docs/docs/uguide/validation.md` describes an "internal validation helper module" that this review removes. `review/suggestions.md` still names `identifier/fork.py`. | **[reading]**  |
| F8  | **The two stable identifiers are two modules with one body.** `uuid5.py` and `content.py` repeat the same three parameters and two 80-line docstrings. The caveat "the collision resistance of the UUID is bounded by SHA-1" is misleading: SHA-1 has a birthday bound of 2^80, a UUID has 122 random bits and a bound of 2^61, so the **width of the UUID** is the limit, not the hash.                                                                                                                                                                                                                                                                                                                                                                | **[reading]**  |
| F9  | **Checksummed identifiers end with a symbol that is awkward to type.** The check symbol is one of the 37 values of Crockford's mod-37 scheme, so **13.6 % of the identifiers end with `*`, `~`, `$`, `=` or `U`** (5 of 37), characters that a shell, a URL or a spreadsheet treats specially, for a feature meant for support codes and license keys. The detection claim of the docstring is true (no single substitution and no adjacent swap was missed in about 740,000 mutations).                                                                                                                                                                                                                                                                | **[verified]** |
| F10 | **`validation.py` is a grab bag with a misleading name.** It holds the Crockford alphabet and decoder (not validation), two range helpers that duplicate `coola.validation`, and it is named like a core package. Its docstring says it "previously duplicated" checks, a history that does not belong in a docstring.                                                                                                                                                                                                                                                                                                                                                                                                                                  | **[reading]**  |
| F11 | **Fork handling is repeated three times.** `MonotonicRandom`, `ObjectIdGenerator` and `SnowflakeIdGenerator` each hand-write a lock, `track_fork` and a `_reset_after_fork` that differs (new random value; new random value and counter; keep the state). There is no module-level default for ULID and UUIDv7 that a test can replace, while ObjectId and Snowflake have a class **and** a default singleton.                                                                                                                                                                                                                                                                                                                                         | **[reading]**  |
| F12 | **Tests: more code than the source, and the wrong properties.** 2,659 lines for 1,861, of which the Snowflake file is 495 (it patches `time.sleep` and `time.time_ns`). 42 uses of `range(...)` in the unit tests (the `CLAUDE.md` rule is no logic in a unit test). The only property tests are about the obfuscated identifiers and import the private `_MASK`, `_encode_base62`, `_decode_base62`. Nothing tests the ordering, the round trip, the shape or the check symbol with Hypothesis. Two tests and `tests/unit/identifier/conftest.py` read the private singleton of another package (`hashing_interface._default_registry.is_built()`) to check that a custom registry does not build the default one.                                     | **[reading]**  |

What is already right and must not regress: the stable identifiers use the frozen registry
(`get_stable_registry()`, 22.4 of `suggestions.md`) and keep their values; the monotonic rule of
`MonotonicRandom` (fresh random in a new tick, increment in the same tick or after a backward step,
borrow the next tick on overflow) is correct and is the one this design generalises; the unbiased
mask-based sampling and the lock around the state; the fork reset through `coola.utils.fork`; 100 %
line coverage of the package; no optional import.

## 4. Architecture

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

### 4.1 The engine: one clock rule

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
  guide (section 1, "Not guaranteed"). For Snowflake it happens above 4,096 identifiers per
  millisecond (4 million per second).
- **The validation of the packed timestamp stays in the format module** (a ULID timestamp is 48 bits,
  a Snowflake one is 41 bits from its epoch), after `next`, so a borrowed tick that leaves the range
  raises a `ValueError` like an explicit one.

### 4.2 The formats

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
  field notices (decision D5).
- **Snowflake** takes its `worker_id` in the constructor (required: no default `0`), owns one
  `TickSequence(bits=12, random_start=False)`, and has no module-level function and no default
  singleton: a process-wide generator has one `worker_id`, which only the caller knows. In a forked
  child, `_reset_after_fork` sets a flag and `generate` raises
  `RuntimeError("SnowflakeIdGenerator cannot be used in a forked process: create one with its own worker_id")`.
  A silent duplicate (101 of 200 forks today, F3) becomes a loud error. `last_timestamp_ms`,
  `sequence`, `_SPIN_SLEEP_S` and the `RuntimeError` for a backward clock are deleted.
- **Crockford** has one decoder: case-insensitive, `O` is `0`, `I` and `L` are `1` (the rule of the
  specification, now also applied to ULID), and `encode(value, length)` is the loop of today's
  `_encode_base32` generalised. `verify_checksummed_id` and `extract_ulid_timestamp_ms` call it.
- **Validation** is `validate_int` then `validate_closed_range` of `coola.validation`, with the
  parameter name, written inline (two lines per parameter). The 48-bit timestamp check of ULID and
  UUIDv7 is repeated in the two modules instead of being a shared private helper: it is two lines,
  and `CLAUDE.md` forbids private helpers imported across modules.

### 4.3 The content-derived identifiers

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

Same computation as today, so **the values do not change** (golden vectors, section 9). One module,
one docstring for the shared parameters, and a corrected note: the UUID is limited by its 122 random
bits (birthday bound 2^61), not by SHA-1; use `generate_stable_content_id` when more bits are needed.

### 4.4 The checksummed identifiers

- `generate_checksummed_id(*, length=12, group_size=4, sep="-")` and
  `verify_checksummed_id(identifier, *, sep="-")`: keyword-only options (F4).
- **The generator never emits the extended check symbols.** If the mod-37 remainder of the payload is
  32 or more, the payload is drawn again (5 of 37 payloads, 1.16 draws on average), so the identifier
  is made of `0-9A-Z` and the separator: safe in a shell, a URL and a spreadsheet (F9). The cost is
  `log2(37/32)`, that is 0.21 bit of the entropy of a 60-bit payload. `verify_checksummed_id` still
  accepts the five symbols of the specification, so identifiers from another implementation verify.
- `sep` must not be in the 32-character alphabet (it is stripped before the check), which is shorter
  to state than "the extended alphabet".
- `verify_checksummed_id` returns `False` and never raises, for a wrong type, a non-ASCII character
  (the dotless `i` upper-cases to `I`) or a wrong alphabet; the rule stays the one of today.

## 5. Public API after the change

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

## 6. Dependents

| Where                                                             | Change                                                                                                                                                                                                         |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/hashing/test_golden.py`, `tests/integration/hashing/` | None: they use the stable identifiers, which keep their values. `test_content.py` gets its own golden vectors for the two identifiers (section 9).                                                             |
| `docs/docs/uguide/identifier.md`, `docs/docs/refs/identifier.md`  | The guide is rewritten around the two families, with the **Properties** table of section 1 at the top, and a decision table "Which one should I use?" that covers every kept name. The reference is unchanged. |
| `README.md`, `docs/docs/index.md`                                 | The blurb "ULID, UUID (v4, v5, v7), Snowflake, ObjectId, NanoID, and prefixed, checksummed, obfuscated and content-based identifiers" becomes the list of the kept names.                                      |
| `docs/docs/dev/architecture.md`                                   | `identifier` leaves the "standalone helpers" (it depends on `hashing`); the layering section says so. New import-linter contract in `pyproject.toml`: `coola.hashing` must not import `coola.identifier`.      |
| `docs/docs/uguide/validation.md`                                  | Remove the sentence about the "internal validation helper module" of `identifier`; it uses `validate_int` and `validate_closed_range` now.                                                                     |
| `docs/docs/migration/v1.1_to_v2.0.md`                             | New section: removed names and their replacements (section 5), Snowflake constructor, ObjectId layout and `timestamp_s`, keyword-only options, stricter extractors. Stable identifiers: unchanged.             |
| `review/suggestions.md`                                           | Replace the reference to `identifier/fork.py` (`coola.utils.fork`), add the status rows of this review.                                                                                                        |

## 7. Evidence: a prototype of the engine

A prototype of the engine and of the four formats (`TickSequence`, the real `coola.utils.fork`, the
current extractors as the oracle, scripts kept outside the repository) was run on `main`. It is not
part of the repository. **[verified]**

| Check                                                                                     | Result                                                              |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Round trip with the **current** extractors (`ulid`, `uuid7`, `object_id`, `snowflake`)    | `extract(generate(timestamp)) == timestamp` for the four            |
| Strictly increasing and unique, 200,000 sequential calls                                  | yes for the four                                                    |
| Unique under 8 threads x 20,000 calls                                                     | yes for the four                                                    |
| Snowflake, 200,000 calls in one loop (more than 4,096 per millisecond)                    | 85 ms, no `RuntimeError`, no sleep, strictly increasing             |
| Base32 encoding of a ULID by the new `encode` against `_encode_base32` on 20,000 payloads | identical                                                           |
| Fork: a parent and a child with the default Snowflake of `main`                           | the same identifier in 101 of 200 trials (the bug the flag removes) |
| Checksummed: 720,000 single substitutions and 19,405 adjacent swaps on random payloads    | 0 undetected                                                        |
| Checksummed: share of the identifiers that end with `*`, `~`, `$`, `=` or `U` on `main`   | 13.6 % (5 of 37)                                                    |

Cost of one call, mean of 50,000 calls (`timeit`, one run each, back to back) on the same machine (Apple M1, Python 3.14.8):

| Generator   | `main`  | Prototype |
| ----------- | ------- | --------- |
| `ulid`      | 2.33 µs | 2.06 µs   |
| `uuid7`     | 1.38 µs | 1.27 µs   |
| `snowflake` | 0.87 µs | 0.37 µs   |
| `object_id` | 0.69 µs | 0.40 µs   |

The gate (section 9) is "no slower than `main`". The reason for the Snowflake gain is that the
prototype drops the validation and the spin-wait bookkeeping of the hot path; it will be a little
slower once `validate_int` is added, still below `main`.

## 8. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Breaking                               |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 1    | **Fix the documentation (F7).** Package docstring, the **Properties** table of section 1 at the top of the guide and in `identifier/__init__.py`, complete the "Which one should I use?" table, fix the architecture sentence and add the import-linter contract. Documentation and `pyproject.toml` only.                                                                                                                                                                | no                                     |
| 2    | **Remove what is not worth owning (F1).** `uuid4.py`, `nanoid.py`, `prefixed.py`, `obfuscated.py` with their tests, the obfuscated property tests, the guide, README and index entries, and the migration section with the replacements of section 5.                                                                                                                                                                                                                     | yes                                    |
| 3    | **One decoder, strict inputs, strict inverses (F5, F6, F10).** `crockford.py` (`ALPHABET`, `encode`, `decode`), delete `validation.py`, `validate_int` and `validate_closed_range` everywhere, strict extractors (`[0-9a-f]{24}`, canonical UUID, `str` only, 63 bits, the first ULID character), `extract_object_id_timestamp_s`. The rejected inputs of F6 become regression tests.                                                                                     | yes (stricter inputs, one name)        |
| 4    | **The engine (F2, F11).** `monotonic.py` becomes `sequence.py` with `TickSequence(bits, tick_ns, random_start)`; ULID and UUIDv7 use it with keyword-only `timestamp_ms`; ObjectId moves to it, `ObjectIdGenerator` and its `get_default_generator` are deleted. Benchmark gate.                                                                                                                                                                                          | yes                                    |
| 5    | **Snowflake (F3).** `SnowflakeIdGenerator(*, worker_id)`, one `TickSequence(bits=12, random_start=False)`, the flag that raises in a forked child, delete `generate_snowflake_id`, `last_timestamp_ms`, `sequence`, the spin-wait, the backward-clock `RuntimeError` and the tests that patch `time.sleep`. Integration test with `os.fork` and `run_in_children`.                                                                                                        | yes                                    |
| 6    | **One content module (F8).** Merge `uuid5.py` into `content.py`; `generate_stable_uuid5` calls `generate_stable_content_id`; fix the SHA-1 note; golden vectors for both identifiers **before** the change, unchanged after.                                                                                                                                                                                                                                              | no                                     |
| 7    | **Checksummed identifiers (F4, F9).** Keyword-only options, rejection sampling so that the generator never emits `*~$=U`, `sep` outside the 32-character alphabet, `verify` that never raises. Property test of the single-substitution and adjacent-swap detection.                                                                                                                                                                                                      | yes (keyword-only, no extended symbol) |
| 8    | **Tests and documentation (F12).** Property tests (section 9), the thread and fork integration tests, replace the two tests and the `conftest.py` fixture that read `hashing_interface._default_registry.is_built()` by the behavioural test that already exists (a hasher registered in the default registry does not change the stable identifier), final pass on the guide, the architecture page, the migration page and the status table of `review/suggestions.md`. | no                                     |

## 9. Tests and gates

Unit tests mirror the source tree (`tests/unit/identifier/`); the conventions of `CLAUDE.md` apply
(one behaviour per test, literal expected values, a regression test for every bug, no local imports,
no loop in a unit test).

- **Engine** (`test_sequence.py`, replaces `test_monotonic.py`), with `time.time_ns` patched:
  a new tick restarts the counter (random: the value comes from `secrets`; `random_start=False`: `0`);
  the same tick increments; a backward clock keeps the tick and increments; an exhausted counter
  (`bits=1`) borrows the next tick and restarts; `tick_ns` divides the clock; two sequences are
  independent. The thread test is in `tests/integration/` without patching.
- **Each format** (`test_ulid.py`, `test_uuid7.py`, `test_objectid.py`, `test_snowflake.py`):
  the shape (length, alphabet, version and variant bits, hex), a **fully specified literal** for an
  explicit `timestamp` with the counter part masked (the timestamp field is asserted exactly), the
  boundaries of the timestamp (`0`, the maximum, the maximum plus one), the types (`True`, `1.5`,
  `"1"` raise `TypeError`), the extractor on every rejected input of F6, and the explicit timestamp
  does not touch the state (the next call of the current time is unchanged).
- **Snowflake**: `worker_id` is required, out of range raises, two generators with two ids never
  collide on the same tick, and the forked-child flag raises (the flag is set by calling
  `_reset_after_fork`, the real fork is in the integration test).
- **Content** (`test_content.py`): **golden vectors** for `generate_stable_content_id` (two lengths)
  and `generate_stable_uuid5` (default and custom namespace), the order of the keys does not matter,
  a custom registry and `ignore_unhashable` are forwarded, and an unhashable object raises
  `KeyError`. The vectors are written **before** step 6 and must be identical after it.
- **Crockford** (`test_crockford.py`): `encode(decode(x))`, case, `O`/`I`/`L`, a character outside
  the alphabet, the empty string, the length of the result.
- **Checksummed**: the shape, no character outside `0-9A-Z` and the separator, the grouping, the
  separators rejected, `verify` for lower case and the confusable letters, `False` for a wrong type,
  non-ASCII, too short, and every wrong-symbol case.
- **Properties** (`tests/properties/test_identifier.py`, rewritten: the obfuscated tests go away with
  their module):
  - round trip `extract(generate(timestamp)) == timestamp` for the four formats;
  - for `t1 < t2`, the identifier of `t1` sorts before the identifier of `t2` (string order for
    ULID, UUIDv7 and ObjectId, numeric for Snowflake), whatever the random part;
  - the shape (a regular expression) of every generated identifier;
  - `verify_checksummed_id(generate_checksummed_id(...))`, any single substitution and any adjacent
    swap of different characters are detected;
  - the stable identifiers of two equal-but-reordered mappings are equal.
- **Integration** (`tests/integration/identifier/`): threads (8 threads x 1,000 calls, with a lock
  around the shared list and the docstring "Test that ...") are unique and, per thread, increasing for
  the four formats; 200,000 sequential Snowflake identifiers (more than 4,096 per millisecond) are
  strictly increasing and never raise; `os.fork`: a child of ULID, UUIDv7 and ObjectId does not
  repeat the identifiers of its parent, a child of a `SnowflakeIdGenerator` raises, a lock held at
  the fork does not block the child (the tests of `test_fork.py`, updated); the stable identifiers
  do not change when a plugin hasher is registered in the default registry.
- **Doctests** run as part of the docstrings; the guide examples are doctests.
- **Benchmark gate** (`tests/benchmarks/`, the CI benchmark): one call of each time-ordered
  generator and one stable identifier of a small dictionary; no slower than `main` (section 7).
- Coverage: 100 % of `src/coola/identifier` (`--cov=coola`), no `# pragma: no cover`.

## 10. Decisions

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
`obfuscated`._ The rule is the one of section 2: a generator stays if it is hard to write correctly
and safe. `uuid4`, `prefixed` and `nanoid` are a line of the standard library (section 5), cost 3
modules (4 with `obfuscated`), and each one has to be documented, tested and kept consistent with the others.
`obfuscated` is the one with a real argument against it: it looks like encryption, leaks the parity,
is broken by two pairs, and needs a 47-line module docstring on its limits; a user who needs an opaque public identifier
should get one from a library made for it. Alternatives: (B) keep everything and apply only steps
1, 3, 4, 5, 6 and 7, with `Callable`-based `prefixed` and the fixes of F4 and F5 for the four modules
(about 420 more lines of source and 480 of tests to maintain), or (C) move the whole package to a
separate distribution and keep only `content.py` in `coola`. (C) is the cleanest layering, and the
most work for the users; it is worth considering if the time-ordered formats keep growing.

**Decided: A.**

**D2. Keep the checksummed identifier?** _Recommended: yes, with rejection sampling._ It is the one
module that shares code with ULID (the decoder), its detection claim is true and tested, and the fix
of F9 makes its output usable for what it is for. If D1(C) is chosen it moves with the others.

**Decided: as recommended.**

**D3. Snowflake without a module-level function.** _Recommended: yes._ A process-wide generator has
one `worker_id`, which the library cannot know; today the id is a per-call argument that shares one
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
shorter than today. The documented MongoDB layout (5-byte per-process value and 3-byte counter) is
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

## 11. Not covered

- **Unpredictable tokens** (sessions, API keys, password reset links): use `secrets.token_urlsafe`;
  none of these identifiers is a capability (section 1).
- **Other formats**: UUIDv6 and v8, KSUID, TypeID, Sqids. The engine makes a new time-ordered format
  a 20-line module; nothing here adds one.
- **Datetime helpers**: `extract_*` return an `int` (milliseconds or seconds since the Unix epoch);
  a `datetime` is one call away and keeping the return type small avoids a time-zone decision.
- **A pluggable clock.** The tests use the explicit timestamp and a patched `time.time_ns` for the
  engine only; a `clock` parameter on every generator is not worth its surface.
- **Allocation of Snowflake worker ids** across hosts (a database row, an environment variable, a
  coordinator): the caller's job, stated in the guide.
- **The format version of the stable identifiers.** A `HASH_FORMAT_VERSION` mixed into the UUID5
  name is item 22.4.3 of `suggestions.md` and belongs to `hashing`.
- **Windows.** `os.register_at_fork` does not exist there; the fork rules do not apply and the
  `os.fork` tests are skipped, as today.

## Appendix: effect on the size of the package

| Module                                                  | Lines today   | Expected after | Why                                                                                |
| ------------------------------------------------------- | ------------- | -------------- | ---------------------------------------------------------------------------------- |
| `__init__.py`                                           | 51            | about 45       | twelve names, the package docstring with the Properties                            |
| `content.py` + `uuid5.py`                               | 80 + 117      | about 110      | one module, one docstring for the shared parameters                                |
| `monotonic.py` -> `sequence.py`                         | 81            | about 70       | the three cases and two parameters, no change of rule                              |
| `validation.py` -> `crockford.py`                       | 87            | about 55       | alphabet, `encode`, `decode`; the range checks use `coola.validation`              |
| `ulid.py`                                               | 139           | about 90       | no `_encode_base32`, strict extractor                                              |
| `uuid7.py`                                              | 133           | about 95       | no duplicated validation, canonical extractor                                      |
| `objectid.py`                                           | 227           | about 65       | no class, no singleton, no lock, no `get_default_generator`                        |
| `snowflake.py`                                          | 346           | about 110      | no spin-wait, no `RuntimeError` clock path, no persistence arguments, no singleton |
| `checksummed.py`                                        | 184           | about 130      | keyword-only, one decoder, rejection sampling in six lines                         |
| `uuid4.py`, `nanoid.py`, `prefixed.py`, `obfuscated.py` | 37+109+74+196 | 0              | removed (D1)                                                                       |
| **Total source**                                        | **1,861**     | **about 770**  |                                                                                    |
| Unit, integration and property tests                    | 2,659         | about 1,200    | fewer modules, no patched `time.sleep`, loops move to Hypothesis                   |

The line counts are estimates from the prototype and from the code to delete; the gate is the
behaviour (sections 1 and 9), not the size.

## Appendix: benchmark of the old implementation (baseline)

`tests/benchmarks/test_identifier_benchmark.py` measures the implementation on `main` (`11a0a8c0`)
and is meant to be run unchanged on the new one. It only uses the public generators and extractors
that exist in both designs (the Snowflake generator is imported with a fallback to
`SnowflakeIdGenerator(worker_id=0).generate`, the constructor of the new design). The time-ordered
generators, the extractors and the checksummed functions are called 10,000 times per round, so the
time divided by 10,000 is the cost of one call. The stable identifiers are measured per call.
Compare with `inv test.benchmark` (or `pytest tests/benchmarks/test_identifier_benchmark.py
--benchmark-only --benchmark-save=<name>`, then `--benchmark-compare=<name>`).

Machine: Apple M1, macOS (Darwin), Python 3.14.8, `pytest-benchmark`. One run, nothing else heavy
running. The absolute numbers depend on the machine; compare the old and the new on the same one.
The raw result is saved in `.benchmarks/` as `identifier_old_11a0a8c0` (the directory is ignored by
git). The "New" columns are to be filled when a step of section 8 lands; the speed-up is old time /
new time (above 1 is faster). "Prototype" is the per-call cost measured in section 7, for the four
generators it covers.

| Benchmark (`test_<name>_benchmark`) | Input                        | Old median | Old min  | Per call (old) | Prototype per call | New median | Speed-up |
| ----------------------------------- | ---------------------------- | ---------- | -------- | -------------- | ------------------ | ---------- | -------- |
| `generate_ulid`                     | 10,000 calls                 | 21.20 ms   | 20.80 ms | 2.12 µs        | 2.06 µs            |            |          |
| `generate_uuid7`                    | 10,000 calls                 | 13.99 ms   | 13.68 ms | 1.40 µs        | 1.27 µs            |            |          |
| `generate_snowflake_id`             | 10,000 calls                 | 5.92 ms    | 5.80 ms  | 0.59 µs        | 0.37 µs            |            |          |
| `generate_object_id`                | 10,000 calls                 | 5.10 ms    | 5.00 ms  | 0.51 µs        | 0.40 µs            |            |          |
| `extract_ulid_timestamp_ms`         | 10,000 calls                 | 25.06 ms   | 24.57 ms | 2.51 µs        |                    |            |          |
| `extract_uuid7_timestamp_ms`        | 10,000 calls                 | 8.15 ms    | 8.07 ms  | 0.81 µs        |                    |            |          |
| `extract_snowflake_timestamp_ms`    | 10,000 calls                 | 1.55 ms    | 1.51 ms  | 0.15 µs        |                    |            |          |
| `generate_checksummed_id`           | 10,000 calls                 | 38.68 ms   | 37.12 ms | 3.87 µs        |                    |            |          |
| `verify_checksummed_id`             | 10,000 calls                 | 18.67 ms   | 18.56 ms | 1.87 µs        |                    |            |          |
| `generate_stable_content_id_small`  | dict of a `str` and an `int` | 5.4 µs     | 5.2 µs   | 5.4 µs         |                    |            |          |
| `generate_stable_uuid5_small`       | dict of a `str` and an `int` | 7.2 µs     | 6.9 µs   | 7.2 µs         |                    |            |          |
| `generate_stable_content_id_nested` | 100 floats, 100 small dicts  | 698.4 µs   | 681.4 µs | 698.4 µs       |                    |            |          |
| `generate_stable_uuid5_nested`      | 100 floats, 100 small dicts  | 692.8 µs   | 683.0 µs | 692.8 µs       |                    |            |          |

Reading the numbers:

- **The costs are small and dominated by encoding, not by the lock.** A UUIDv7 costs 1.4 µs and a
  ULID 2.1 µs: the difference is the 26-iteration Base32 loop of `_encode_base32`, and the ULID
  extractor pays the same loop in reverse (2.5 µs against 0.8 µs for UUIDv7). The gate is "no
  slower than old" for every row.
- **The stable identifiers cost what `hash_object` costs.** The UUID5 wrapper adds 1.8 µs to a small
  input (one `uuid.uuid5`), nothing measurable on the nested one. Merging the two modules (step 6)
  must not change these rows.
- **The checksummed generator is the slowest row** (3.9 µs, 38.7 ms per 10,000): the mod-37 decode of
  the payload is a Python loop over the characters. The rejection sampling of step 7 draws a payload
  again in 5 of 37 cases (1.16 draws on average), so expect about 1.16 times this cost for the
  generator, and no change for `verify_checksummed_id`; the shared `crockford.decode` of step 3
  should keep both rows within 10 % of the old numbers.
- **Not in the table:** the behaviour that no benchmark of speed can show (strictly increasing
  values, the Snowflake burst above 4,096 identifiers per millisecond, which today sleeps and
  raises on a backward clock) is covered by the integration tests of section 9.
