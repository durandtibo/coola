# Architecture of `coola.summary`

:book: This page describes the design of the `coola.summary` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/summary.md#properties); the API is in the [reference](../refs/summary.md).

## Overview

`summarize` uses one engine (`coola.summary.engine`), one extension point and one option object
(`SummaryConfig`). A summarizer only describes one type: `summarize(data, config, *, compact)`
returns a `str` for a leaf, or a `Node` (a header, the lazy `(label, child)` pairs and their number)
for a container. The engine owns the invariants: the summarizer lookup (by MRO), the display depth,
`max_items` and `max_characters`, the cycle check, the layout, and the errors (a summarizer that
raises is replaced by a placeholder). It walks the nodes with an explicit stack, so there is no
`RecursionError`. A container at the display depth is shown on one line by `CompactRepr`, whose work
is bounded by `max_items`, so the cost depends on the size of the summary and not on the size of the
data. This is the same split as the equality and the recursive engines.

## Constraints

1. **Simple and easy to maintain.** The package is 2,397 lines and 4,020 lines of unit tests for one
   function. The target removes the code that exists only to work around the design (the three
   `_get_preview`, the translation of `RecursionError`, the constructors, `__repr__` and `equal` of
   every summarizer, the three base classes of the metadata summarizers) instead of adding code next
   to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. The deprecated positional options are removed. The release notes
   list the changes, including the new layout of the text.
3. **Not slower where it matters.** `summarize` is called to build a log line. Small inputs (tens of
   microseconds) may cost a little more for the fixed cost of the engine, but a huge input must not
   cost more than a small one.
4. **Consistent with the siblings.** Same engine shape as `equality` (the engine owns the invariants,
   a plugin describes one type), same registry, same option rules as `recursive` and `equality`
   (keyword-only, validated, one frozen config).
5. **Optional libraries stay lazy.** The package imports no optional library with
   `import coola.summary`; the summarizers of NumPy, torch, pandas, ... are loaded the first time the
   registry needs a type of the library (`register_lazy`, as before). The plugin group
   `coola.summary.summarizers` is kept.

## Architecture in detail

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

### The summarizer contract

```python
@dataclass(frozen=True, slots=True)
class Node:
    header: str  # the first line: "<class 'list'> (length=3)"
    children: Iterable[tuple[Any, Any]]  # lazy (label, child) pairs, in display order
    total: int  # the number of children, to announce "... (+N more)"


class Summarizer(Protocol[T_contra]):
    def summarize(self, data: T_contra, config: SummaryConfig, *, compact: bool) -> str | Node:
        """A string (a leaf) or a ``Node`` (a container)."""


class BaseSummarizer(ABC, Generic[T]):  # convenience base class, as before
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

### The config

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

### The engine

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
  several lines is aligned under its first character. Nothing is re-indented.
- **A failure is local.** A node that fails prints its type and the exception name (never the message,
  which may carry data) and the rest of the summary is intact.

### The compact form

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
  and `safe_str(data)` of .
- Any other type (a subclass such as `Counter`, a dataclass, an array) is shown with its `repr`, cut
  at `max_characters`. It is computed in full, which is the documented exception of "Bounded work".
- It is built only when a container is at the boundary (lazy), so a flat summary pays nothing.
- Fast path (required by the benchmark of the appendix): a container with at most `max_items` scalar
  entries is shown with `repr()` directly, because `reprlib` is slower than `repr` on small
  containers.

The alternative that was considered, "header only" (`<class 'list'> (length=100) ...`), is bounded
and needs no `reprlib`, but it hides the content of a small nested configuration, which is the most
common use of the default `max_depth=1`.

### The layout of the text

Rules (the examples are the output of the prototype):

```
>>> from coola.summary import summarize
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
  .
- **The type is shown for the nodes that are expanded and for their leaves; the boundary shows
  values.** A leaf shows `<class 'int'> 1` when `compact` is `False` and `1` when it is `True`. This
  is the previous rule, kept: `summarize(42)` is `<class 'int'> 42`, and the items of
  `summarize([1, "a"])` are bare because they are at the display depth.
- A metadata summarizer always shows its type: `<class 'numpy.ndarray'> | shape=(3,) | dtype=float64`.
- Indentation is a fixed 2 spaces per level.

Differences with the previous version, to list in the release notes: the nested levels are indented by 2 (not 4),
the marker of the hidden entries is `... (+N more)` (was `...`), an empty container is
`(length=0)` (was `<class 'list'> []`), a list at the boundary is `[0, 1, 2, 3, 4, ...]` (was
`[0, 1, 2, 3, 4] ...`), a dict at the boundary keeps its order but a `Counter` or other subclass is
its `repr`, and `num_spaces` is gone.

### Built-in summarizers

| Summarizer                                                                                                 | `summarize(data, config, compact=...)`                                                            | Default registration                  |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `DefaultSummarizer`                                                                                        | `safe_str(data)`, preceded by `<class ...> ` unless `compact`; a leaf                             | `object`, `UserString`                |
| `SequenceSummarizer`                                                                                       | `Node("<type> (length=n)", enumerate(data), n)` (the field names for a named tuple)               | `list`, `tuple`, `deque`, `Sequence`  |
| `SetSummarizer`                                                                                            | `Node("<type> (length=n)", enumerate(smallest(data, config.max_items)), n)`                       | `set`, `frozenset`                    |
| `MappingSummarizer`                                                                                        | `Node("<type> (length=n)", data.items(), n)`                                                      | `dict`, `Mapping`, `MappingProxyType` |
| `DataclassSummarizer`                                                                                      | `Node("<type> (fields=n)", ((f.name, getattr(data, f.name)) for f in fields(data) if f.repr), n)` | `DataclassMarker`                     |
| `PydanticModelSummarizer`                                                                                  | the same with `model_fields` without the secrets, plus `model_extra`                              | `pydantic.BaseModel` (lazy)           |
| `NDArraySummarizer`, `TensorSummarizer`, `JaxArraySummarizer`, `Pandas*`, `Polars*`, `Pyarrow*`, `Xarray*` | `" \| ".join([type, *self._describe(data, config)])`, or `safe_repr(data)` if `config.show_data`  | by library (lazy), as before          |

Container summarizers are three one-line classes on a common helper; the dataclass and pydantic
summarizers share `BaseObjectFieldsSummarizer` (one abstract method `_get_fields`). All of them are
stateless: no constructor, no `__repr__`, no `equal`, no `InlineDisplayMixin`.

The metadata base class keeps one job: `BaseMetadataSummarizer` (`_describe(data, config)` returns the
fields, `_check_dependencies` raises `MissingPackageError` at construction if the library is
missing). `BaseFieldsSummarizer` is deleted: the summarizers of frames and tables call
`format_fields(items, config.max_items)` (`a: int64, b: float64, ... (+3 more)`) themselves.

Removed from the default registry: `str`, `int`, `float`, `complex`, `bool` (resolved through `object`,
the comment of `_get_native_summarizers` says so). Added: `UserString` and `MappingProxyType`.
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

## Public API

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

## Dependents

Nothing in `src` imports `coola.summary` (the mentions in `equality` and `utils.singleton` are
docstrings). The package is a leaf, which is why the contract can change in one step.

| Dependent                                                                                        | Change                                                                                                                                          |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/properties/test_summary_random.py`, `test_cross_module`, `test_library_types`             | Keep, add the properties of. They use only `summarize(data, max_depth=...)`.                                                                    |
| `tests/integration/test_lazy_imports.py`, `test_support_matrix.py`, `test_default_registries.py` | Keep (lazy loading, the support matrix). Update the expected registered types (`UserString`, `MappingProxyType`).                               |
| `pyproject.toml` import-linter contracts                                                         | Unchanged: `coola.summary` still imports only `coola.registry`, `coola.utils`, `coola.validation`.                                              |
| Plugins of `coola.summary.summarizers`                                                           | Must be ported to the new `summarize(data, config, *, compact)` (migration page). A plugin that returned a string only needs the new signature. |

`SummarizerRegistry.find_summarizer` copies the `DataclassMarker` fallback that `EqualityTesterRegistry`,
`HasherRegistry`, `TransformerRegistry` and `ChildFinderRegistry` also copy. Moving it to
`BaseTypeDispatchRegistry` is a cleanup for all five packages, outside this design.

## Decisions

All ten decisions were taken as recommended. The design above is final on these points; the text of
each decision is kept for the reasoning and for the alternatives that were rejected.

**D1. Options in a config, not in the constructors?** _Decided: yes (recommended)._ A person who wants three
items writes `summarize(data, max_items=3)`; before they build a registry, register `object` and `list`,
and lose the other types. The summarizers become stateless, so `__init__`, `__repr__`, `equal`
and the display mixin disappear, and the package matches `EqualityConfig`. Cost: no per-type
option (`max_items=3` for lists and 10 for dicts); the user who needs it registers their own
summarizer (a one-line subclass that returns a `Node` with a smaller `total`).

Decision: **accepted as recommended.**

**D2. One limit for items, fields and columns?** _Decided (recommended): yes, `max_items` (default 5)._
`MAX_FIELDS=10` and `max_items=5` count the same thing (entries shown). Consequence: a frame lists 5
columns by default instead of 10; `summarize(df, max_items=20)` lists more. Alternative: keep
`max_fields` as a second option, or default `max_items` to 10 (a list of 20 shows 10 lines).

Decision: **accepted as recommended.**

**D3. Keep the name `max_depth`?** _Decided: yes (recommended)._ It is a display depth here and a limit that
raises in `objects_are_equal`; the docs say so in both places. Renaming to
`expand_depth` removes the ambiguity, but `max_depth` is the name every user of `summarize` knows, and
the cost of the ambiguity is a documentation sentence.

Decision: **accepted as recommended.**

**D4. What is the compact form?** _Decided (recommended): a `reprlib`-based one line (4.4)._ It keeps the content
of a small nested structure, which is the point of the default `max_depth=1`, and it is bounded.
Alternatives (rejected): the header only (simplest, hides small configs); asking the summarizer for an
inline form (`Node.inline`), which is exact for every type but doubles the contract.

Decision: **accepted as recommended.** Consequence: the benchmark appendix shows that `reprlib` makes
`tree[1]` 2.5x slower than the previous version, so must meet the gate of with a fast path in
`CompactRepr` (`repr()` directly when the container has at most `max_items` scalar entries) and
re-measure. The gate stays "up to 2x and at most 10 µs more" for inputs under 50 µs; relaxing it is not
part of the decision.

**D5. Total summarizing?** _Decided (recommended): yes, catch `Exception` in the engine._ A summary is for logs
and error messages; raising there hides the original problem. The cost is that a bug in a custom
summarizer shows as `<summary failed: ValueError>` instead of a traceback; the exception type is in
the text and the tests of the plugin show it. Alternative: let the exceptions propagate and only keep
`safe_str`.

Decision: **accepted as recommended.**

**D6. Hide the `field(repr=False)` fields of a dataclass?** _Decided: yes (recommended)._ It is how a dataclass
author declares "do not print this" (tokens, large buffers), and the pydantic summarizer already hides
secrets. The summary is not a comparison, so unlike `equality` and `hashing` it does not need every
field. Alternative: show them (today), and document that a dataclass is printed in full.

Decision: **accepted as recommended.**

**D7. Fixed indentation?** _Decided (recommended): yes, 2 spaces, no `num_spaces`._ The option exists to fix a
layout that was irregular; a regular layout does not need it. Adding it back is one field of the
config.

Decision: **accepted as recommended.**

**D8. Keep the rule "type shown inside the expansion, values at the boundary"?** _Decided (recommended): keep._
It is the documented behaviour of `summarize(42)` and of every example, and it needs one flag
(`compact`). Alternative: always show the type of a leaf (`(a): <class 'int'> 1` everywhere): a
simpler rule, but twice as long on the default output.

Decision: **accepted as recommended.**

**D9. One protocol or two?** _Decided (recommended): keep `Summarizer` (protocol) and `BaseSummarizer` (ABC)._
Same as `Hasher`/`BaseHasher` and `Transformer`/`BaseTransformer`; the cost is a few lines and the
benefit is that a summarizer from another library needs no import of `coola`.

Decision: **accepted as recommended.**

**D10. Named tuple labels?** _Decided: yes (recommended)._ One line in `SequenceSummarizer` (`data._fields`),
consistent with dataclasses. Alternative: positions only (today).

Decision: **accepted as recommended.**

## Not covered

- **Plain objects with `__dict__`, `attrs` classes, `__slots__`:** leaves, as before. `equality` has
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
  (the `recursive` design).
- The entry-point plugin group `coola.summary.summarizers` keeps its name; a plugin that implements
  `summarize(data, registry, depth, max_depth)` must be ported (migration page).
