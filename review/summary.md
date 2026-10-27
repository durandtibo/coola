# `coola.summary`: target state and the changes to reach it

This document describes the design that `coola.summary` should have, then the changes to make to the
current implementation (`main`, `fbd13317`) to get there. The design follows the rules of the user
guide (`docs/docs/uguide/summary.md`, **Properties** and **Design Principles**) and the architecture
guide (`docs/docs/dev/architecture.md`: registry dispatch, MRO lookup, lazy imports, strict
contracts), and it stays aligned with the sibling packages `equality`, `recursive` and `iterator`
(`review/equality.md`, `review/recursive.md`). Its aim is a small package that is easy to maintain:
**one engine, one extension point, one option object**. Breaking changes are accepted, and the future
state is more important than the past one.

Evidence comes from probes and a throw-away prototype run on `main` (Python 3.14, scripts kept
outside the repository). Items marked **[verified]** were reproduced; **[reading]** comes from the
code. Nothing in `src`, `tests` or `docs` was changed.

## 1. Properties of the sub-package

A summary is a short text that a person reads in a log, a notebook or an error message, to know what
a (possibly huge, nested, partly unprintable) object contains. These are the rules of the package.
Every design choice below exists to make one of them true **by construction** (in the engine)
instead of by the discipline of each summarizer.

| Property                   | Statement                                                                                                                                                                                                                                                                                                                                                                         | Enforced by                                 |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Bounded output             | A node shows at most `max_items` children (items, fields, columns), a text is cut at `max_characters`, and the expansion stops at `max_depth`. What is not shown is announced (`... (+N more)`), never silently dropped.                                                                                                                                                          | the engine                                  |
| Bounded work               | The cost depends on the size of the output, not on the size of the data: a collection is never iterated beyond `max_items`, a node below `max_depth` is never expanded, and a sub-structure that is not shown is not converted to a string. Listed exceptions: `len()`, sets (the smallest items are selected in one pass), and the `repr` of an object that has no bounded form. | the engine, lazy children, the compact form |
| Display depth              | `max_depth` is the number of levels that are expanded. Deeper nodes are shown on **one line** (the compact form). It never raises, unlike the `max_depth` of `objects_are_equal` (a limit).                                                                                                                                                                                       | the engine                                  |
| Total                      | Summarizing never fails because of the data: a `__str__`, `__repr__`, `__len__` or attribute that raises is replaced by a placeholder that names the exception type. Invalid options raise `ValueError` before any work.                                                                                                                                                          | the engine                                  |
| Cycle-safe                 | A container that contains itself is shown as `<cycle>` where it comes back, whatever the summarizer registered for it.                                                                                                                                                                                                                                                            | the engine                                  |
| Any depth                  | No `RecursionError`, whatever `max_depth` or the caller's stack: the engine uses an explicit stack. The size of the output is bounded by `max_depth`, `max_items` and `max_characters`.                                                                                                                                                                                           | the engine                                  |
| Type dispatch              | The summarizer is chosen from `type(data)` along the Method Resolution Order, so the most specific registered type wins. A summarizer is always found: `object` is registered by the default registry.                                                                                                                                                                            | the registry                                |
| Registry-extensible        | A new type is supported by registering one summarizer, without changing another one and without subclassing a base class (a protocol is enough).                                                                                                                                                                                                                                  | the registry, the `Summarizer` protocol     |
| Metadata first             | An array, a tensor, a data frame, a table shows its metadata (type, shape, dtype, columns), not its values. `show_data=True` shows a bounded `repr` instead. A lazy frame is never collected.                                                                                                                                                                                     | the metadata summarizers                    |
| Secrets stay hidden        | The fields that are marked as secret (`SecretStr`, `SecretBytes`) or excluded from the `repr` (`field(repr=False)`) are not shown.                                                                                                                                                                                                                                                | the object summarizers                      |
| Iterators are not consumed | A generator or an iterator is a leaf: its `str` is shown and no item is taken from it.                                                                                                                                                                                                                                                                                            | the default registry                        |
| Deterministic              | The same value gives the same text across calls and processes: the order of a list, a tuple or a dict is the order of the data, and a set shows its smallest items in order (it does not depend on `PYTHONHASHSEED`). Only the address in a placeholder or in an unknown `repr` can change.                                                                                       | the set summarizer                          |
| Pure                       | The data is never modified. Summarizers keep no state: a registry, a config and a summarizer can be shared by threads and calls.                                                                                                                                                                                                                                                  | the engine, stateless summarizers           |
| Consistent with siblings   | Dataclasses are entered (like `objects_are_equal`, `hash_object`, `recursive_apply`); iterators are leaves (like `dfs_iterate`); options are keyword-only.                                                                                                                                                                                                                        | the default registry                        |
| For humans                 | The output is meant to be read, not parsed: its exact format can change in a release (this design changes it, section 4.5).                                                                                                                                                                                                                                                       | (documented)                                |

Not guaranteed, and documented as such:

- **Shared references.** A container reachable by several paths (a DAG) is shown once per path (only
  a cycle is cut). The output stays bounded by `max_items` and `max_depth`.
- **The bound on a chain.** Each level is indented, so a chain of `n` levels costs `O(n²)` characters
  (`max_depth=50,000` on a 50,000-level list gives 2.5 GB in the prototype **[verified]**). `max_depth`
  is what bounds the output; the user who asks for a huge depth gets a huge text.
- **The time of a foreign `str` or `repr`.** The package bounds what it iterates, not what a user
  class does inside `__str__` or `__repr__` (they are computed in full and then cut).
- **A stable format** between releases (see "For humans").

## 2. Constraints

1. **Simple and easy to maintain.** The package is 2,397 lines and 4,020 lines of unit tests for one
   function. The target removes the code that exists only to work around the design (the three
   `_get_preview`, the translation of `RecursionError`, the constructors, `__repr__` and `equal` of
   every summarizer, the three base classes of the metadata summarizers) instead of adding code next
   to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. The deprecated positional options are removed. The release notes
   list the changes, including the new layout of the text (section 4.5).
3. **Not slower where it matters.** `summarize` is called to build a log line. Small inputs (tens of
   microseconds) may cost a little more for the fixed cost of the engine, but a huge input must not
   cost more than a small one (section 9, gates).
4. **Consistent with the siblings.** Same engine shape as `equality` (the engine owns the invariants,
   a plugin describes one type), same registry, same option rules as `recursive` and `equality`
   (keyword-only, validated, one frozen config).
5. **Optional libraries stay lazy.** The package imports no optional library with
   `import coola.summary`; the summarizers of NumPy, torch, pandas, ... are loaded the first time the
   registry needs a type of the library (`register_lazy`, as today). The plugin group
   `coola.summary.summarizers` is kept.

## 3. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Status         |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The recursion lives in each summarizer.** Every container summarizer calls `registry.summarize(child, depth + 1, max_depth)`. The depth is a public argument of `Summarizer.summarize` and of `registry.summarize` ("internal use"), and the protections (the translation of `RecursionError` in `SummarizerRegistry.summarize`) work only if the summarizer cooperates. A nested list raises `RecursionError` with `max_depth` between 300 and 400 (a message that tells to reduce the nesting). `equality` and `bfs_iterate` have no such limit. | **[verified]** |
| F2  | **The compact form is not bounded.** At the display depth, a container with at most `max_items` entries is shown with `safe_str(data)`: the whole sub-structure is converted to a string, and then cut at 1,000 characters. `summarize([[list(range(2_000_000))] * 5])`, which prints 1,036 characters, takes 0.27 s. The preview of a container that is larger than the limit is rebuilt by three `_get_preview` methods (and `_first_items`) so that the type is kept in the text.                                                                 | **[verified]** |
| F3  | **The options live in the constructors of the summarizers.** `max_items`, `num_spaces`, `max_characters`, `show_data` and `max_fields` are arguments of `SequenceSummarizer(...)`, `DefaultSummarizer(...)`, `NDArraySummarizer(...)`. To show 3 items the user builds a registry, registers `object` and `list`, and loses the other types (the guide spends 90 lines on it). Every summarizer needs `__init__`, `__repr__` (`InlineDisplayMixin`), `_display_fields` and `equal`; none of it carries information.                                  | **[reading]**  |
| F4  | **A secret is printed.** A dataclass field declared `field(repr=False)` (the standard way to hide a token) is listed by `DataclassSummarizer`: `summarize(User("a"))` shows `(token): SECRET`. The pydantic summarizer hides `SecretStr` fields, so the two object summarizers disagree.                                                                                                                                                                                                                                                             | **[verified]** |
| F5  | **Wrong dispatch for two standard types.** `collections.UserString` inherits from `Sequence`, so it is shown as a list of characters (`(0): x`); `str` is a leaf. `types.MappingProxyType` is only a virtual `Mapping`, so it falls to `DefaultSummarizer` and is printed whole (`recursive_apply` registers both).                                                                                                                                                                                                                                  | **[verified]** |
| F6  | **The layout is not regular.** The indentation grows by 2 for the first level and by 4 for the next ones (`num_spaces=2` gives 2, 6, 10 spaces), because each level re-indents the text of its children (`O(depth × size)`). A value with a newline is not aligned (`(0): <class 'str'> a` / `    b`). The guide says `num_spaces` is "the number of spaces for indenting each level".                                                                                                                                                               | **[verified]** |
| F7  | **Not total.** A `__len__` that raises (`class L(list)`), a dataclass with a missing attribute, a property that raises: `summarize` raises (`RuntimeError`, `AttributeError`). Only `str(data)` is guarded (`safe_str`). A log call must not crash the job.                                                                                                                                                                                                                                                                                          | **[verified]** |
| F8  | **A cycle is repeated up to `max_depth`.** The guide documents "self-referencing containers are not detected": the output repeats the cycle until the display depth, and fails with F1 for a large depth. `equality`, `hashing` and `recursive` detect cycles.                                                                                                                                                                                                                                                                                       | **[verified]** |
| F9  | **Deprecated and loose options.** `summarize(data, 2)` still works through `deprecate_positional_options`. `register_summarizers(mapping, exist_ok=False)` takes `exist_ok` positionally. `max_depth=True` and `max_depth=1.5` are accepted. `max_items=-1` and `max_characters=-1` mean "no limit", which the guide documents only for the first.                                                                                                                                                                                                   | **[verified]** |
| F10 | **Three base classes for the metadata summarizers.** `BaseSummarizer` → `BaseMetadataSummarizer` (`show_data`) → `BaseFieldsSummarizer` (`max_fields`), plus `MAX_FIELDS` and `format_fields` exported. The only reason for the third level is that `max_fields` is a constructor argument (F3). `MAX_FIELDS` (10) and `max_items` (5) are two limits for the same idea.                                                                                                                                                                             | **[reading]**  |
| F11 | **`ARG002` noqa.** Four `# noqa: ARG002` (`DefaultSummarizer`, `BaseMetadataSummarizer`) exist because the signature passes `registry`, `depth` and `max_depth` to summarizers that need none of them (`review/noqa-review.md`).                                                                                                                                                                                                                                                                                                                     | **[reading]**  |
| F12 | **The guide contradicts the code.** "Comparing Data Structures" says that with `max_depth=1` nested structures "are shown as raw strings without truncation" (they are cut at `max_items`: `[0, 1, 2, 3, 4] ...`); "Not guaranteed: self-referencing containers are not detected" is a finding (F8); "Recursive summarization" is a design principle that is an implementation detail; the `Raises: RecursionError` of `summarize` documents F1.                                                                                                     | **[verified]** |
| F13 | **A named tuple has no names.** `summarize(Point(1, 2), max_depth=2)` prints `(0)`, `(1)`; a dataclass prints `(x)`, `(y)`.                                                                                                                                                                                                                                                                                                                                                                                                                          | **[verified]** |

What is already right and must not regress: type dispatch with MRO and a cache; the lazy loading of
the summarizers of optional libraries; metadata instead of values for arrays, tensors and frames;
`PolarsLazyFrameSummarizer` never collects the data (the schema only); the secret fields of pydantic
models; the deterministic set preview (`PYTHONHASHSEED`); `safe_str` for a broken `__str__`; iterators
and generators as leaves; the display depth that does not raise.

## 4. Architecture

```
summarize(data, *, max_depth=1, max_items=5, max_characters=1000, show_data=False, registry=None)
        |   one SummaryConfig (frozen, validated), one registry (default: the singleton)
        v
render(data, registry, config)                coola/summary/engine.py   (about 70 lines)
        |   explicit stack of frames; ids of the nodes being expanded; lines are appended once
        |
        |-- summarizer = registry.find_summarizer(type(obj))            MRO, DataclassMarker
        |-- result     = summarizer.summarize(obj, config, compact=...)  pure, no recursion
        |        str                        -> a leaf: the text, cut at max_characters
        |        Node(header, children, n)  -> a container:
        |              compact (depth >= max_depth)  one bounded line (CompactRepr)
        |              already open (cycle)          "<cycle>"
        |              otherwise                     header, then at most max_items children,
        |                                            then "... (+N more)"
        |-- an exception of the summarizer or of the children iterator -> a placeholder line
        v
the text (lines joined once)
```

The engine owns the invariants (bounded output, bounded work, depth, cycles, totality, layout). A
summarizer **only describes one type**: a string for a leaf, or a header and lazy children for a
container. That is the same split as `equality` (the engine owns identity, type, depth and cycles; a
tester returns a verdict or the child pairs) and as `recursive` (a transformer returns the children).
`registry`, `depth` and `max_depth` are no longer arguments of the summarizer, so a summarizer
cannot forget a protection and cannot recurse.

### 4.1 The summarizer contract

```python
@dataclass(frozen=True, slots=True)
class Node:
    header: str  # the first line: "<class 'list'> (length=3)"
    children: Iterable[tuple[Any, Any]]  # lazy (label, child) pairs, in display order
    total: int  # the number of children, to announce "... (+N more)"


class Summarizer(Protocol[T_contra]):
    def summarize(self, data: T_contra, config: SummaryConfig, *, compact: bool) -> str | Node:
        """A string (a leaf) or a ``Node`` (a container)."""


class BaseSummarizer(ABC, Generic[T]):  # convenience base class, as today
    @abstractmethod
    def summarize(self, data: T, config: SummaryConfig, *, compact: bool) -> str | Node: ...

    def equal(self, other: object) -> bool:  # concrete: same type (all the built-ins are stateless)
        return type(other) is type(self)
```

Rules of the contract (documented in `BaseSummarizer`, checked by one parametrised unit test over the
default registry, like `NATIVE_SAMPLES` in `equality`):

1. `summarize` is pure and cheap: it does not call the registry, does not recurse, does not modify
   `data`, and does not iterate the whole data. For a container it reads `len(data)` and returns the
   children **lazily** (`enumerate(data)`, `data.items()`, a generator): the engine takes at most
   `max_items` of them. A set is the exception: it returns the smallest `max_items` items.
2. `compact` is `True` when the node is at the display depth. A leaf may use it (the default summarizer
   shows the type only when `compact` is `False`); a container ignores it (the engine uses
   `CompactRepr`).
3. A leaf returns a `str`, which may have several lines. The engine cuts it at `max_characters`, so a
   leaf does not have to.
4. Labels are anything: the engine writes `safe_str(label)` cut at `max_characters`.
5. An exception (of `summarize`, of the children iterator, of a label's `str`) is the engine's
   problem: it prints a placeholder, the summarizer does not guard itself.

`Node` is a dataclass and not a tuple so that the contract is readable and a later field (for example
a closing line) can be added without breaking the plugins.

### 4.2 The config

```python
@dataclass(frozen=True, slots=True)
class SummaryConfig:  # shareable, hashable, thread-safe; validated in __post_init__
    max_depth: int = 1  # non-negative int, not a bool
    max_items: int | None = 5  # non-negative int, or None for no limit
    max_characters: int | None = 1000  # non-negative int, or None for no limit
    show_data: bool = False  # metadata summarizers show a bounded repr of the values
```

All the options of the package are here, in one place, with one validation. `None` replaces the
negative sentinel (`-1` was "no limit" for `max_items` and `max_characters` but `max_depth=-1` was an
error). `num_spaces` and `max_fields` are deleted (decisions D2, D7).

### 4.3 The engine

```python
INDENT = "  "


def render(data, registry, config):
    lines: list[str] = []
    ancestors: set[int] = set()  # ids of the nodes being expanded
    stack: list[Frame] = []  # Frame: children iterator, depth, total, shown, id

    def cut(text):  # the character limit, for a leaf and for a label
        n = config.max_characters
        return text if n is None or len(text) <= n else text[:n] + "..."

    def visit(label, obj, depth):
        compact = depth >= config.max_depth
        prefix = INDENT * depth
        head = prefix if label is None else f"{prefix}({cut(safe_str(label))}): "
        try:
            result = registry.find_summarizer(type(obj)).summarize(obj, config, compact=compact)
            if not isinstance(result, Node):
                text = cut(result)
            elif compact:
                text = CompactRepr(config).repr(obj)  # one bounded line
            elif id(obj) in ancestors:
                text = "<cycle>"
            else:
                text = result.header
                ancestors.add(id(obj))
                children = islice(result.children, config.max_items)
                stack.append(Frame(children, depth + 1, result.total, 0, id(obj)))
        except Exception as exc:  # the data is broken, not the summary
            text = f"{type(obj)} <summary failed: {type(exc).__name__}>"
        lines.append(head + text.replace("\n", "\n" + prefix + INDENT))

    visit(None, data, 0)
    while stack:
        frame = stack[-1]
        try:
            label, child = next(frame.children)
        except StopIteration:
            stack.pop()
            ancestors.discard(frame.id)
            if frame.total > frame.shown:
                lines.append(f"{INDENT * frame.depth}... (+{frame.total - frame.shown:,} more)")
        except Exception as exc:  # a property that raises, a broken __iter__
            frame.children = iter(())
            lines.append(f"{INDENT * frame.depth}<summary failed: {type(exc).__name__}>")
        else:
            frame.shown += 1
            visit(label, child, frame.depth)
    return "\n".join(lines)
```

Design points:

- **No `RecursionError`, no thread-local, no lock.** The state (stack, ancestors, lines) is local to
  the call. `raise_recursion_limit_error` is not used by the package any more, and the `RecursionError`
  paragraphs of the docstrings and of the guide are deleted.
- **`summarize` of a container runs at the boundary too.** It is cheap by contract (rule 1), and it
  keeps the engine free of a "is this a container?" question that only the summarizer can answer.
- **The cycle check is on the open nodes only** (`ancestors`), so a DAG is not reported as a cycle, and
  a leaf (a string, an array) never enters the set.
- **The layout is one rule**: a node at depth `d` is written with `2 * d` spaces, and a text with
  several lines is aligned under its first character. Nothing is re-indented (F6).
- **A failure is local.** A node that fails prints its type and the exception name (never the message,
  which may carry data) and the rest of the summary is intact.

### 4.4 The compact form

The compact form is what the engine prints for a container at the display depth: **one line, with
bounded work**. It is a `reprlib.Repr` configured from the config and with two overrides:

```python
class CompactRepr(reprlib.Repr):
    def __init__(self, config):
        super().__init__()
        self.maxlevel = 3
        self.maxlist = self.maxtuple = self.maxdict = self.maxset = ... = config.max_items
        self.maxstring = self.maxother = config.max_characters

    def repr_dict(self, x, level):  # keeps the insertion order: reprlib sorts the keys
        ...

    def repr_set(self, x, level):  # the smallest items, as SetSummarizer (reprlib sorts everything)
        ...
```

- `list`, `tuple`, `deque`, `dict`, `set`, `frozenset` and `array` are cut at `max_items`
  (`[0, 1, 2, 3, 4, ...]`) in `O(max_items)`, nested containers are cut at 3 levels (`[...]`), and a
  string is shortened in the middle (`'abc...xyz'`). This replaces `_get_preview` ×3, `_first_items`
  and `safe_str(data)` of F2.
- Any other type (a subclass such as `Counter`, a dataclass, an array) is shown with its `repr`, cut
  at `max_characters`. It is computed in full, which is the documented exception of "Bounded work".
- It is built only when a container is at the boundary (lazy), so a flat summary pays nothing.

The alternative that was considered, "header only" (`<class 'list'> (length=100) ...`), is bounded
and needs no `reprlib`, but it hides the content of a small nested configuration, which is the most
common use of the default `max_depth=1` (decision D4).

### 4.5 The layout of the text

Rules (the examples are the output of the prototype of section 7):

```
>>> print(summarize({"a": [1, 2, 3, 4, 5, 6, 7], "b": {"c": {1, 2}, "d": "x\ny"}}, max_depth=3))
<class 'dict'> (length=2)
  (a): <class 'list'> (length=7)
    (0): <class 'int'> 1
    (1): <class 'int'> 2
    (2): <class 'int'> 3
    (3): <class 'int'> 4
    (4): <class 'int'> 5
    ... (+2 more)
  (b): <class 'dict'> (length=2)
    (c): <class 'set'> (length=2)
      (0): 1
      (1): 2
    (d): <class 'str'> x
      y
>>> print(summarize([1, "a", [1, 2]]))           # max_depth=1: the items are on one line
<class 'list'> (length=3)
  (0): 1
  (1): a
  (2): [1, 2]
>>> a = [1]; a.append(a); print(summarize(a, max_depth=3))
<class 'list'> (length=2)
  (0): <class 'int'> 1
  (1): <cycle>
```

- A node is `header`, then its children one level deeper, then `... (+N more)` if some are not shown.
  An empty container is its header alone: `<class 'list'> (length=0)`.
- A child is `(label): text`. The label of a sequence or a set is the position, of a mapping the `str`
  of the key, of a dataclass or a pydantic model the field name, of a named tuple the field name
  (F13).
- **The type is shown for the nodes that are expanded and for their leaves; the boundary shows
  values.** A leaf shows `<class 'int'> 1` when `compact` is `False` and `1` when it is `True`. This
  is today's rule, kept (decision D8): `summarize(42)` is `<class 'int'> 42`, and the items of
  `summarize([1, "a"])` are bare because they are at the display depth.
- A metadata summarizer always shows its type: `<class 'numpy.ndarray'> | shape=(3,) | dtype=float64`.
- Indentation is a fixed 2 spaces per level.

Differences with `main`, to list in the release notes: the nested levels are indented by 2 (not 4),
the marker of the hidden entries is `... (+N more)` (was `...`), an empty container is
`(length=0)` (was `<class 'list'> []`), a list at the boundary is `[0, 1, 2, 3, 4, ...]` (was
`[0, 1, 2, 3, 4] ...`), a dict at the boundary keeps its order but a `Counter` or other subclass is
its `repr`, and `num_spaces` is gone.

### 4.6 Built-in summarizers

| Summarizer                                                                                                 | `summarize(data, config, compact=...)`                                                            | Default registration                  |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `DefaultSummarizer`                                                                                        | `safe_str(data)`, preceded by `<class ...> ` unless `compact`; a leaf                             | `object`, `UserString`                |
| `SequenceSummarizer`                                                                                       | `Node("<type> (length=n)", enumerate(data), n)` (the field names for a named tuple)               | `list`, `tuple`, `deque`, `Sequence`  |
| `SetSummarizer`                                                                                            | `Node("<type> (length=n)", enumerate(smallest(data, config.max_items)), n)`                       | `set`, `frozenset`                    |
| `MappingSummarizer`                                                                                        | `Node("<type> (length=n)", data.items(), n)`                                                      | `dict`, `Mapping`, `MappingProxyType` |
| `DataclassSummarizer`                                                                                      | `Node("<type> (fields=n)", ((f.name, getattr(data, f.name)) for f in fields(data) if f.repr), n)` | `DataclassMarker`                     |
| `PydanticModelSummarizer`                                                                                  | the same with `model_fields` without the secrets, plus `model_extra`                              | `pydantic.BaseModel` (lazy)           |
| `NDArraySummarizer`, `TensorSummarizer`, `JaxArraySummarizer`, `Pandas*`, `Polars*`, `Pyarrow*`, `Xarray*` | `" \| ".join([type, *self._describe(data, config)])`, or `safe_repr(data)` if `config.show_data`  | by library (lazy), as today           |

Container summarizers are three one-line classes on a common helper; the dataclass and pydantic
summarizers share `BaseObjectFieldsSummarizer` (one abstract method `_get_fields`). All of them are
stateless: no constructor, no `__repr__`, no `equal`, no `InlineDisplayMixin`.

The metadata base class keeps one job: `BaseMetadataSummarizer` (`_describe(data, config)` returns the
fields, `_check_dependencies` raises `MissingPackageError` at construction if the library is
missing). `BaseFieldsSummarizer` is deleted: the summarizers of frames and tables call
`format_fields(items, config.max_items)` (`a: int64, b: float64, ... (+3 more)`) themselves.

Removed from the default registry: `str`, `int`, `float`, `complex`, `bool` (resolved through `object`,
the comment of `_get_native_summarizers` says so). Added: `UserString` and `MappingProxyType` (F5).
Kept as they are: `get_default_registry`, `get_default_summarizer_registry` (alias),
`register_summarizers`, the plugin group.

The extension example of the guide becomes:

```python
class PointSummarizer(BaseSummarizer[Point]):
    def summarize(self, data, config, *, compact):
        return f"Point(x={data.x}, y={data.y})"  # a leaf


class NodeSummarizer(BaseSummarizer[Tree]):
    def summarize(self, data, config, *, compact):  # a container
        return Node(
            f"{type(data)} (children={len(data.children)})",
            enumerate(data.children),
            len(data.children),
        )


register_summarizers({Point: PointSummarizer(), Tree: NodeSummarizer()})
```

## 5. Public API after the change

```python
from coola.summary import (
    summarize,  # (data, *, max_depth=1, max_items=5, max_characters=1000, show_data=False, registry=None)
    SummaryConfig,
    SummarizerRegistry,  # find_summarizer, has_summarizer, summarize(data, config=None)
    get_default_registry,
    get_default_summarizer_registry,  # alias, unchanged
    register_summarizers,  # (mapping, *, exist_ok=False)
    Summarizer,  # protocol: summarize(data, config, *, compact) -> str | Node
    BaseSummarizer,
    Node,
    DefaultSummarizer,
    SequenceSummarizer,
    SetSummarizer,
    MappingSummarizer,
    DataclassSummarizer,
    BaseMetadataSummarizer,
    format_fields,  # (fields, max_fields) -> str
    # lazy: NDArraySummarizer, TensorSummarizer, JaxArraySummarizer, PandasDataFrameSummarizer,
    # PandasSeriesSummarizer, PolarsDataFrameSummarizer, PolarsLazyFrameSummarizer,
    # PolarsSeriesSummarizer, PyarrowArraySummarizer, PyarrowTableSummarizer,
    # XarrayDataArraySummarizer, XarrayDatasetSummarizer, XarrayVariableSummarizer,
    # PydanticModelSummarizer
)
```

Removed: `BaseCollectionSummarizer`, `BaseFieldsSummarizer`, `MAX_FIELDS`, the positional options of
`summarize`, the `depth` and `max_depth` arguments of `SummarizerRegistry.summarize` and of
`Summarizer.summarize`, the constructor arguments of every summarizer, `RecursionError` from the
documented exceptions.

`summarize(...)` builds a `SummaryConfig` (so the options are validated in one place) and calls
`registry.summarize(data, config)`, which calls the engine; `registry.summarize` stays the
registry-level entry point, as `registry.objects_are_equal` is for `equality`.

## 6. Dependents

Nothing in `src` imports `coola.summary` (the mentions in `equality` and `utils.singleton` are
docstrings). The package is a leaf, which is why the contract can change in one step.

| Dependent                                                                                        | Change                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/properties/test_summary_random.py`, `test_cross_module`, `test_library_types`             | Keep, add the properties of section 9. They use only `summarize(data, max_depth=...)`.                                                                                     |
| `tests/integration/test_lazy_imports.py`, `test_support_matrix.py`, `test_default_registries.py` | Keep (lazy loading, the support matrix). Update the expected registered types (`UserString`, `MappingProxyType`).                                                          |
| `docs/docs/uguide/summary.md`, `dev/architecture.md`, `migration/v0.11_to_v1.0.md`               | Rewritten in step 7. The "Custom max_items", "Customizing Indentation" and "Working with NumPy and PyTorch" sections become one-liners with `summarize(..., max_items=3)`. |
| `pyproject.toml` import-linter contracts                                                         | Unchanged: `coola.summary` still imports only `coola.registry`, `coola.utils`, `coola.validation`.                                                                         |
| Plugins of `coola.summary.summarizers`                                                           | Must be ported to the new `summarize(data, config, *, compact)` (migration page). A plugin that returned a string only needs the new signature.                            |

`SummarizerRegistry.find_summarizer` copies the `DataclassMarker` fallback that `EqualityTesterRegistry`,
`HasherRegistry`, `TransformerRegistry` and `ChildFinderRegistry` also copy. Moving it to
`BaseTypeDispatchRegistry` is a cleanup for all five packages, outside this design (section 11).

## 7. Evidence: a prototype of the engine

A prototype (about 110 lines: `Node`, a config, the loop of 4.3, a `reprlib` subclass for the compact
form, the three container summarizers and the dataclass one) was run on `main`. It is not part of the
repository. **[verified]**

| Check                                                                | `main`                                 | Prototype                       |
| -------------------------------------------------------------------- | -------------------------------------- | ------------------------------- |
| a list nested 50,000 levels, `max_depth=10**6`                       | `RecursionError` (at about 300 to 400) | works (output of 2.5 GB, see 1) |
| `summarize([[list(range(2_000_000))] * 5])` (compact form)           | 0.27 s                                 | 0.0001 s (143 characters)       |
| a self-referencing list, `max_depth=3`                               | repeats the cycle                      | `<cycle>`                       |
| a `field(repr=False)` field                                          | printed                                | hidden                          |
| a tree (20 dicts of lists, a dict of 100 ints, a set), `max_depth=3` | 45.0 µs                                | 35.1 µs                         |
| the same tree, `max_depth=1`                                         | 15.5 µs                                | 26.1 µs                         |
| `summarize(1)`                                                       | 0.87 µs                                | 1.30 µs                         |
| `summarize([])`                                                      | 0.94 µs                                | 1.87 µs                         |

The prototype is slower for the smallest inputs, and not tuned: it builds the config and the
`reprlib` object on every call (the real engine builds `CompactRepr` lazily, only for a container at
the boundary, and the config once per call in `summarize`). The fixed cost is a few microseconds; the
gain is on the inputs where `main` is not bounded (F1, F2). The gate of section 9 is stated in these
terms.

## 8. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Breaking                    |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------- |
| 1    | **Fix the guide (F12).** Remove "raw strings without truncation", replace "not detected" by the current behaviour, add the **Properties** table of section 1 at the top of the page and in the package docstring. Add `tests/benchmarks/test_summary_benchmark.py` and record the baseline of `main` (section 9). Documentation and benchmark only.                                                                                                                                                                                              | no                          |
| 2    | **Fix the dispatch (F5).** Register `UserString` (leaf) and `MappingProxyType` (mapping); drop the redundant `str`/`int`/`float`/`complex`/`bool` entries. Regression tests that fail on `main`.                                                                                                                                                                                                                                                                                                                                                 | no (output of two types)    |
| 3    | **Remove the deprecated and loose options (F9).** Delete `deprecate_positional_options` on `summarize`, make `exist_ok` of `register_summarizers` keyword-only, validate `max_depth` (int, not bool).                                                                                                                                                                                                                                                                                                                                            | yes (positional options)    |
| 4    | **Engine, `Node` and the new contract (F1, F2, F6, F7, F8, F11, F13).** Add `engine.py` (`render`, `CompactRepr`), `Node`; change `Summarizer`/`BaseSummarizer` to `summarize(data, config, *, compact)`; port the container, dataclass, pydantic and metadata summarizers; `SummarizerRegistry.summarize(data, config=None)` calls the engine; delete the recursion translation, `_get_preview`, `_first_items`, the `ARG002` noqa. In this step `config` carries only `max_depth` and the old constructor options still exist. Benchmark gate. | yes (contract, text layout) |
| 5    | **Options move to `SummaryConfig` (F3, F9, F10).** `max_items`, `max_characters`, `show_data` become keyword options of `summarize`; delete the constructors, `__repr__`, `_display_fields`, `InlineDisplayMixin`, the `equal` overrides, `BaseCollectionSummarizer`, `BaseFieldsSummarizer`, `MAX_FIELDS`, `num_spaces`; `None` replaces the negative sentinel; `format_fields(fields, max_fields)`.                                                                                                                                            | yes                         |
| 6    | **Secrets (F4).** `DataclassSummarizer` skips the `field(repr=False)` fields. Regression test.                                                                                                                                                                                                                                                                                                                                                                                                                                                   | yes (output)                |
| 7    | **Documentation.** Rewrite `uguide/summary.md` (properties, options as arguments, the compact form, the extension example), the `summary` paragraph of `dev/architecture.md`, the migration page (`migration/v0.11_to_v1.0.md`: the new signature, the options, the layout), the `Properties` of the package docstring, and the status table of `review/suggestions.md` with the PR numbers.                                                                                                                                                     | no                          |

Steps 2, 3 and 6 are independent and can be merged in any order before 4. Step 5 can be merged into
4 if the diff stays reviewable.

## 9. Tests and gates

Unit tests mirror the source tree (`tests/unit/summary/`); the conventions of `CLAUDE.md` apply (one
behaviour per test, literal expected values, the whole output asserted against a literal at least
once, a regression test for every bug, no local imports, the `*_available` fixtures). The 4,020 lines
of today shrink: the tests of constructors, `__repr__`, `equal`, `num_spaces`, `_get_preview` and the
negative sentinels disappear with the code.

- **Engine** (`test_engine.py`, new):
  - the full text of a nested list/dict/tuple/set/dataclass against a literal, for `max_depth` 0, 1,
    2 and 3 (the layout of 4.5);
  - `max_items=0`, `1`, exactly the length, the length plus one, `None`: the `... (+N more)` line and
    its count;
  - `max_characters` on a leaf, on a label and on a header-less text; `None` = no cut;
  - a text with a newline is aligned at every depth; a key with a newline is cut to one label;
  - cycle through a list, a dict, a tuple-in-list, a dataclass: `<cycle>`; a DAG (`[s, s]`) is **not**
    a cycle and is shown twice;
  - depth 100,000 with `max_depth=10**6` does not raise (regression for F1; check it fails on `main`
    with `git stash push src`);
  - bounded work: a `Spy` mapping/sequence that counts the items taken proves that at most
    `max_items` are consumed, and a container at the boundary does not call `str` on its children
    (regression for F2);
  - totality: `__len__`, `__str__`, `__repr__`, a property, `_describe` and a children iterator that
    raise give a placeholder and the rest of the summary is intact; the message of the exception is
    not in the text (F7);
  - an iterator/generator is a leaf and is not consumed;
  - re-entrant call (a `__str__` that calls `summarize`) and a custom registry are independent;
  - a plugin summarizer that returns a `str` or a `Node` (protocol only, no base class).
- **`SummaryConfig`** (`test_config.py`, new): defaults, each field validated (negative, `bool`,
  `float`, `str`), `None` accepted for `max_items` and `max_characters`, frozen, hashable.
- **Compact form** (`test_compact.py`, new): each builtin container cut at `max_items`, insertion
  order of a dict, the smallest items of a set, 3 levels then `[...]`, a string shortened in the
  middle, a subclass shown with its `repr` cut at `max_characters`, a `Counter`.
- **Each summarizer** (`test_<module>.py`): `summarize` asserts the whole result against a literal
  (`Node` header, the list of children, the total), and the typical, empty, single-element and
  duplicate cases; a subclass of `list`/`dict`/`set` keeps its type in the header; named tuple labels;
  `OrderedDict`, `Counter`, `deque`, `UserList`, `UserString`, `MappingProxyType`; a set of mixed
  types; the input is not modified. Dataclass: frozen, slots, `field(repr=False)`, `InitVar`, nested.
  Pydantic: secrets, extras, nested models. Metadata summarizers: `show_data` False and True, a
  masked array, a lazy frame that is not collected, an invalid lazy frame.
- **Contract test over the default registry**: every summarizer of the default registry (including
  the lazy ones when the library is installed) returns a `str` or a `Node` with `total >= 0` and lazy
  `children` (one sample per registered type, like `NATIVE_SAMPLES`).
- **Properties** (`tests/properties/test_summary_random.py`, extend): the number of lines is at most
  `1 + max_items * (1 + ...)` for the given depth (bounded output); the summary of a nested random
  structure is the same twice (kept); the data is not modified (kept); with `max_items=None` and
  `max_characters=None` nothing is hidden (no `...`).
- **Integration** (`tests/integration/summary/`, no patching): threads running `summarize` on the
  default registry (`run_threads`, a lock around the shared list, docstring "Test that ...");
  `PYTHONHASHSEED` 0 and 1 in two subprocesses give the same text for a set of strings; the real
  libraries (NumPy, torch, pandas, polars, pyarrow, xarray, JAX, pydantic) in one nested structure;
  `import coola.summary` does not import any optional library (kept in `test_lazy_imports`).
- **Doctests** run as part of the docstrings; the guide examples are doctests.
- **Benchmark gate** (`tests/benchmarks/test_summary_benchmark.py`, recorded on `main` in step 1):
  scalar, empty list, small dict, the tree of section 7 at `max_depth` 1 and 3, a list of 100,000
  items, five lists of 2,000,000 items at the boundary, and a list nested 300 levels. Gate: every
  input under 50 µs on `main` may cost up to 2x (and at most 10 µs more); every other input must not
  be slower than `main`; the two huge inputs must be at least 100x faster. A failing gate rejects the
  step.
- Coverage: 100 % of `src/coola/summary` (`--cov=coola`), no `# pragma: no cover`.

## 10. Decisions for the maintainers

**D1. Options in a config, not in the constructors?** _Recommended: yes._ A person who wants three
items writes `summarize(data, max_items=3)`; today they build a registry, register `object` and `list`,
and lose the other types (F3). The summarizers become stateless, so `__init__`, `__repr__`, `equal`
and the display mixin disappear, and the package matches `EqualityConfig`. Cost: no per-type
option (`max_items=3` for lists and 10 for dicts); the user who needs it registers their own
summarizer (a one-line subclass that returns a `Node` with a smaller `total`).

**D2. One limit for items, fields and columns?** _Recommended: yes, `max_items` (default 5)._
`MAX_FIELDS=10` and `max_items=5` count the same thing (entries shown). Consequence: a frame lists 5
columns by default instead of 10; `summarize(df, max_items=20)` lists more. Alternative: keep
`max_fields` as a second option, or default `max_items` to 10 (a list of 20 shows 10 lines).

**D3. Keep the name `max_depth`?** _Recommended: yes._ It is a display depth here and a limit that
raises in `objects_are_equal` (28.3 of `suggestions.md`); the docs say so in both places. Renaming to
`expand_depth` removes the ambiguity, but `max_depth` is the name every user of `summarize` knows, and
the cost of the ambiguity is a documentation sentence.

**D4. What is the compact form?** _Recommended: a `reprlib`-based one line (4.4)._ It keeps the content
of a small nested structure, which is the point of the default `max_depth=1`, and it is bounded.
Alternatives: the header only (simplest, hides small configs); asking the summarizer for an inline form
(`Node.inline`), which is exact for every type but doubles the contract.

**D5. Total summarizing?** _Recommended: yes, catch `Exception` in the engine._ A summary is for logs
and error messages; raising there hides the original problem. The cost is that a bug in a custom
summarizer shows as `<summary failed: ValueError>` instead of a traceback; the exception type is in
the text and the tests of the plugin show it. Alternative: let the exceptions propagate and only keep
`safe_str`.

**D6. Hide the `field(repr=False)` fields of a dataclass?** _Recommended: yes._ It is how a dataclass
author declares "do not print this" (tokens, large buffers), and the pydantic summarizer already hides
secrets. The summary is not a comparison, so unlike `equality` and `hashing` it does not need every
field. Alternative: show them (today), and document that a dataclass is printed in full.

**D7. Fixed indentation?** _Recommended: yes, 2 spaces, no `num_spaces`._ The option exists to fix a
layout that was irregular (F6); a regular layout does not need it. Adding it back is one field of the
config.

**D8. Keep the rule "type shown inside the expansion, values at the boundary"?** _Recommended: keep._
It is the documented behaviour of `summarize(42)` and of every example, and it needs one flag
(`compact`). Alternative: always show the type of a leaf (`(a): <class 'int'> 1` everywhere): a
simpler rule, but twice as long on the default output.

**D9. One protocol or two?** _Recommended: keep `Summarizer` (protocol) and `BaseSummarizer` (ABC)._
Same as `Hasher`/`BaseHasher` and `Transformer`/`BaseTransformer`; the cost is a few lines and the
benefit is that a summarizer from another library needs no import of `coola`.

**D10. Named tuple labels?** _Recommended: yes._ One line in `SequenceSummarizer` (`data._fields`),
consistent with dataclasses. Alternative: positions only (today).

## 11. Not covered

- **Plain objects with `__dict__`, `attrs` classes, `__slots__`:** leaves, as today. `equality` has
  `ObjectAttributesEqualityTester`; a summarizer of the attributes is a separate item.
- **A machine-readable form** (JSON, a tree of `Node`): the package produces text for humans. A
  `summary_tree(...)` that returns the nodes is possible later, since the engine already walks them.
- **Width-aware or colored output**, a `str.format`-style template, and options per type.
- **Summaries of the elements of an array** (min, max, mean, number of NaN): a metadata summarizer
  can compute them for the types that want it; the default stays shape and dtype (cheap, no copy,
  no device transfer).
- **The `DataclassMarker` fallback** that five registries copy: it belongs in
  `BaseTypeDispatchRegistry` (one PR for the five packages).
- **`dfs_iterate` and `recursive_apply` explicit stacks:** same technique, separate designs
  (`review/recursive.md`).
- The entry-point plugin group `coola.summary.summarizers` keeps its name; a plugin that implements
  `summarize(data, registry, depth, max_depth)` must be ported (migration page).

## Appendix: effect on the size of the package

| Module                                                           | Lines today                                | Expected after                         | Why                                                                                |
| ---------------------------------------------------------------- | ------------------------------------------ | -------------------------------------- | ---------------------------------------------------------------------------------- |
| `engine.py`                                                      | 0                                          | about 110                              | new: `render`, `CompactRepr`                                                       |
| `config.py`                                                      | 0                                          | about 45                               | new: `SummaryConfig` and its validation                                            |
| `base.py`                                                        | 153                                        | about 90                               | `Node`, the protocol and the ABC; shorter signatures                               |
| `registry.py`                                                    | 242                                        | about 130                              | no `RecursionError` translation, no `depth`/`max_depth`; `summarize(data, config)` |
| `interface.py`                                                   | 245                                        | about 170                              | no positional shim, `summarize` builds the config                                  |
| `collection.py`, `sequence.py`, `mapping.py`, `set.py`           | 76 + 75 + 92 + 141 = 384                   | about 110 (one module, `container.py`) | no `BaseCollectionSummarizer` skeleton, no `_get_preview`, no `_first_items` copy  |
| `object_fields.py`, `dataclass.py`, `pydantic.py`                | 86 + 45 + 77 = 208                         | about 150                              | a `Node` instead of a hand-written layout                                          |
| `default.py`                                                     | 64                                         | about 30                               | no constructor, no `_summarize`                                                    |
| `metadata.py`                                                    | 202                                        | about 90                               | no `BaseFieldsSummarizer`, no `MAX_FIELDS`, no display mixin, no `equal`           |
| `numpy`, `torch`, `jax`, `pandas`, `polars`, `pyarrow`, `xarray` | 95 + 96 + 74 + 110 + 153 + 116 + 155 = 799 | about 700                              | no `ARG002`, `_describe(data, config)`; docstrings are most of the lines           |
| `__init__.py`                                                    | 100                                        | about 90                               | fewer exports                                                                      |
| **Total**                                                        | **2,397**                                  | **about 1,700**                        |                                                                                    |

The line counts are estimates from the prototype and from the code to delete; the gate is the
behaviour (sections 1 and 9), not the size. The unit tests are expected to go from 4,020 to about
3,000 lines.

## Appendix: benchmark of the old implementation and of the prototype

Comparison of `tests/benchmarks/test_summary_benchmark.py` (new file, 15 cases) on `main` and on the
prototype of section 7. **The new design is not implemented yet**, so the "new" column is the
throw-away prototype (about 110 lines: the engine of 4.3, `CompactRepr` built lazily, the three
container summarizers and the dataclass one; no metadata, pydantic or plugin support). It is a lower
bound on the work of the real engine, not a measurement of it; the numbers must be redone with the
real package at step 4 (the gate of section 9). The benchmarks only use `summarize(data,
max_depth=...)` with the default registry, so the file runs unchanged on both.

Machine: Apple M1, macOS (Darwin), Python 3.14, `pytest-benchmark`. Old: `main` at `2900b97a`; new:
the prototype, injected by a pytest plugin that replaces `coola.summary.summarize` (kept outside the
repository). One run each, back to back, nothing else heavy running. The speed-up is old time / new
time: above 1 is faster, below 1 is slower. The absolute numbers depend on the machine.

The raw results are saved in `.benchmarks/Darwin-CPython-3.14-64bit/` (ignored by git):
`0006_summary_old.json` and `0007_summary_proto.json`. Compare them with
`pytest --benchmark-compare=0006 --benchmark-only`.

| Benchmark (`test_summarize_<name>_benchmark`) | Input                                                             | Old median | New median | Speed-up | Old min  | New min  | Speed-up (min) |
| --------------------------------------------- | ----------------------------------------------------------------- | ---------- | ---------- | -------- | -------- | -------- | -------------- |
| `scalar`                                      | `1`                                                               | 1.0 µs     | 1.1 µs     | 0.85x    | 0.9 µs   | 1.0 µs   | 0.87x          |
| `empty_list`                                  | `[]`                                                              | 1.0 µs     | 1.7 µs     | 0.60x    | 0.9 µs   | 1.5 µs   | 0.57x          |
| `small_dict`                                  | `{"a": 1, "b": [2, 3], "c": {"d": 4}}`                            | 5.0 µs     | 7.0 µs     | 0.71x    | 4.8 µs   | 6.8 µs   | 0.72x          |
| `dataclass`                                   | dataclass of a `str`, a list and a dict, `max_depth=2`            | 12.3 µs    | 8.1 µs     | 1.52x    | 12.0 µs  | 7.8 µs   | 1.53x          |
| `tree[1]`                                     | 20 dicts of lists, a dict of 100 ints, a set; `max_depth=1`       | 10.9 µs    | 27.0 µs    | 0.40x    | 10.6 µs  | 26.4 µs  | 0.40x          |
| `tree[3]`                                     | the same, `max_depth=3`                                           | 44.5 µs    | 35.8 µs    | 1.24x    | 43.7 µs  | 35.0 µs  | 1.25x          |
| `flat_list[1]`                                | list of 100,000 ints, `max_depth=1`                               | 5.6 µs     | 4.5 µs     | 1.25x    | 5.4 µs   | 4.3 µs   | 1.25x          |
| `flat_list[2]`                                | list of 100,000 ints, `max_depth=2`                               | 6.6 µs     | 5.4 µs     | 1.22x    | 6.4 µs   | 5.2 µs   | 1.22x          |
| `flat_dict`                                   | dict of 100,000 ints, `max_depth=2`                               | 6.8 µs     | 5.5 µs     | 1.24x    | 6.6 µs   | 5.2 µs   | 1.25x          |
| `flat_set`                                    | set of 100,000 ints, `max_depth=2`                                | 1.08 ms    | 1.08 ms    | 1.01x    | 1.08 ms  | 1.07 ms  | 1.00x          |
| `big_children_at_the_display_depth`           | 5 lists of 200,000 ints at the display depth, `max_depth=1`       | 22.47 ms   | 14.9 µs    | 1510x    | 22.23 ms | 14.5 µs  | 1537x          |
| `deep_list[10]`                               | list nested 10 times, `max_depth=11`                              | 27.9 µs    | 14.7 µs    | 1.89x    | 27.3 µs  | 14.3 µs  | 1.91x          |
| `deep_list[100]`                              | list nested 100 times, `max_depth=101`                            | 1.98 ms    | 139.5 µs   | 14.2x    | 1.95 ms  | 137.3 µs | 14.2x          |
| `deep_list[300]`                              | list nested 300 times, `max_depth=301`                            | 31.40 ms   | 431.4 µs   | 72.8x    | 31.21 ms | 424.7 µs | 73.5x          |
| `tree_expanded`                               | balanced tree of dicts and lists, depth 8, width 3, `max_depth=8` | 25.07 ms   | 7.98 ms    | 3.14x    | 24.59 ms | 7.91 ms  | 3.11x          |

Reading the numbers:

- **Where `main` is not bounded, the design is far faster.** Five big lists at the display depth cost
  22 ms on `main` (it converts all of them to a string, F2) and 15 µs in the prototype, about
  1,500 times less, and independent of the size of the lists. A deep chain is quadratic on `main` (the
  levels re-indent the text of their children, F6): 100 levels 14x faster, 300 levels 73x faster. The
  expanded tree is 3.1x faster for the same reason. A chain of 400 levels or more cannot be measured on
  `main` (F1); the prototype handles it.
- **Flat large inputs are the same or a little faster** (1.2x). Both only touch `max_items` entries.
  The set is the exception by design: the smallest items are selected in one pass over the 100,000
  elements (1.08 ms on both), the documented exception of "Bounded work".
- **The smallest inputs are slower, by a fixed cost of 0.1 to 2 µs** (the scalar 0.85x, the empty list
  0.60x, the small dict 0.71x): the engine builds a frame and a list of lines for a node that `main`
  returns directly. The absolute difference is under 2 µs and does not grow with the input.
- **`tree[1]` is the one regression that matters: 2.5x slower (10.9 µs to 27 µs).** At `max_depth=1`
  the 3 children of the dict are containers shown in the compact form, and `reprlib` is slower than
  `str()` of a small container (it formats item by item, with a level counter). The gate of section 9
  ("up to 2x, at most 10 µs more, for an input under 50 µs") is **not met** by the prototype on this
  case (+16 µs). It is the cost of bounding the compact form (F2) and it only shows on a mid-size
  structure whose children are small. Options before step 4: `CompactRepr` calls `repr()` directly when
  `len(x) <= max_items` and the children are scalars (the common small case), or the gate is relaxed
  to 3x for the compact form (decision D4 lists the alternatives). The real engine must be measured
  again; the prototype is not tuned (it uses `reprlib` as is, one `Frame` list per node, no `__slots__`).
- **These numbers do not include** the metadata summarizers (arrays, frames), the pydantic summarizer,
  the cache of the registry lookup or the validation of the config; none of them is on the hot path of
  the cases above, but each adds a fixed cost per call that the real benchmark will show.
