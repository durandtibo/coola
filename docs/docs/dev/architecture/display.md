# Architecture of `coola.display`

:book: This page describes the design of the `coola.display` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../../uguide/display.md#properties); the API is in the [reference](../../refs/display.md).

## Constraints

1. **Simple and easy to maintain.** The package is 981 lines of `src`, most of them docstrings.
   Remove what walks two structures in parallel
   (`_mask_nested_secret_fields`) and what only exists for one caller; do not add layers.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias,
   no deprecation period. The release notes list the changes.
3. **`rich` and `colorlog` are optional (the point of this review).** `import coola`, `import
coola.display` and `import coola.display.rich` must work without them and must not import them.
   Calling a function that needs a missing library raises `MissingPackageError` with the
   `pip install coola[rich]` hint, not a bare `ModuleNotFoundError`. The tests of `coola.display.rich`
   are skipped when `rich` is not installed instead of failing at collection.
4. **Core package.** `coola.display` sits in the core group (`registry`, `utils`, `validation`,
   `formatting`): it imports no feature package (`lint-imports` contract "Core packages must not
   import feature packages"). It may import `utils`, `validation` and `formatting`.
5. **Consistent with the siblings.** Options are keyword-only (`CLAUDE.md`), arguments are validated
   with `coola.validation`, helpers that exist elsewhere are reused (`coola.formatting`), and the
   optional libraries follow the pattern of `coola.utils.fallback`.

## Architecture

```
coola/display/
    __init__.py       BaseDisplayMixin, InlineDisplayMixin, MultilineDisplayMixin (eager, no dependency)
                      str_pydantic_model, repr_pydantic_model                     (lazy, pydantic)
    mixin.py          the three mixins
    pydantic.py       format a pydantic model without its secrets                   imports pydantic
    colorlog.py       configure_colorlog_logging                                    imports colorlog
    rich/
        __init__.py   the five public names, lazy                                  imports nothing
        logging.py    configure_rich_logging                                        imports rich
        print_.py     print_markdown, print_pretty                                  imports rich
        progressbar.py make_progressbar, make_spinner                               imports rich
coola/utils/fallback/rich.py   stand-ins for the names of rich that the three modules import   (new)
```

The package has three kinds of content with one rule each:

| Kind                | Dependency         | Rule                                                                                                                                |
| ------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| Mixins              | none               | imported eagerly; used by other packages.                                                                                           |
| Pydantic formatting | `pydantic`         | the functions are exported lazily from `coola.display`; the module guards its import.                                               |
| Terminal output     | `rich`, `colorlog` | `coola.display.rich` exports lazily like `coola.display`; each module guards its import; a function never runs without its library. |

### The rule for an optional library (new for `rich`)

The rule is the one of `pydantic.py` and `colorlog.py`, applied to `rich` and written once:

1. **The package `__init__` exports lazily.** `coola/display/rich/__init__.py` uses `lazy_exports`
   (as `coola/display/__init__.py` does for the pydantic functions). `import coola.display.rich`
   imports no module of `rich`; `from coola.display.rich import print_pretty` imports
   `coola.display.rich.print_`.
2. **A module guards its import** with the fallback of `coola.utils.fallback`:

   ```python
   if TYPE_CHECKING or is_rich_available():  # pragma: no cover
       from rich import get_console
       from rich.markdown import Markdown
       ...
   else:  # pragma: no cover
       from coola.utils.fallback.rich import Markdown, Panel, Pretty, get_console, ...
   ```

   `import coola.display.rich.print_` therefore never fails.

3. **The fallback fails at the first use**, with `MissingPackageError("`rich`package is required but
not installed. Please install it with`pip install coola[rich]`")` (the message of
   `raise_package_missing_error`). `coola/utils/fallback/rich.py` (about 25 lines, built with
   `make_fake_class`, `make_fake_function`) defines `get_console`, `RichHandler`, `Markdown`,
   `Panel`, `Pretty`, `Progress` and the six columns as fakes and `MINIMAL`, `ROUNDED` as `None`.
   Every public function calls a fake in its first statement (`Pretty(data)`, `SpinnerColumn()`,
   `get_console()`), before it changes anything: `configure_rich_logging` does not touch the root
   logger when `rich` is missing.
4. **`check_rich()` is not called.** The fallback is the single mechanism, as for `pydantic`; a second
   check would be dead code. (`check_rich` stays in `coola.utils.imports` for the other packages.)
5. **`colorlog` keeps its design**: the function works without the library and falls back to plain
   `logging.basicConfig`, because for a logging helper "no colour" is a correct result and not an
   error. This is documented as the difference between the two logging functions: `rich` is
   required, `colorlog` is optional.

Why the fallback and not an import inside each function: it is the convention of the repository
(`reducer`, `rounding`, `utils.tensor`, `display.pydantic`, `display.colorlog`), the names stay
at the top of the module where `ruff` and `ty` see them, and the tests need no patching of
`builtins.__import__`. The cost is one new file of 25 lines.

### The mixins (unchanged)

`BaseDisplayMixin` defines `__repr__`, `__str__` and a default `_display_fields`; the two concrete
mixins implement `_render_fields`. Nothing changes except the guide. One remark kept for the
maintainers: `MultilineDisplayMixin` uses `repr_indent(repr_mapping(fields))`, which indents the
continuation lines of a multiline value (`(y): a\n  b`). This is intended and covered by a test.

### Formatting a pydantic model (simplified)

The function formats the **fields of the model**, not a dump of it. One private function builds the
mapping, the two public functions choose `repr` or `str`:

```python
def _format_model(model, *, line_fn, sort, exclude_none, exclude_secret, exclude_fields) -> str:
    fields = {name: getattr(model, name) for name, info in type(model).model_fields.items()
              if info.repr}                                   # Field(repr=False) is honoured
    fields |= model.model_extra or {}                         # extra="allow" values
    if exclude_secret:
        for name in secret_field_names(model): fields.pop(name, None)
    ...exclude_none, exclude_fields (warn on unknown names), sort...
    return f"{type(model).__qualname__}({line_fn({k: _shown(v) for k, v in fields.items()})})"
```

with one small recursive helper for the values:

```python
def _shown(value):
    """A nested model is formatted like the root; lists, tuples, sets and dicts are rebuilt."""
```

- **Nested models keep their name** (`P(c=C(name=a), items=[C(name=b)])`, ). A nested model is
  formatted with the same options as the root (`sort`, `exclude_none`, `exclude_secret`); the
  shown value is a tiny object whose `__repr__`/`__str__` is the formatted string, so the generic
  `repr_mapping_line` and `str_mapping_line` of `coola.formatting` need no change.
- **No parallel walk, no `cast`**: the value is read once, from the model; a nested secret is
  dropped by the recursive call on the nested model, so the 45 lines of `_mask_*` and the
  `zip(strict=False)` disappear (about 20 lines for `_shown`).
- **No serializer runs.** `getattr` reads the attribute; a `field_serializer`, an alias or a
  `computed_field` has no effect (a `computed_field` is not a field, so it is not shown; pydantic's
  `repr` shows it, the function does not: it is a property that may be expensive). This is stated
  in the docstring.
- **`secret_field_names` and `is_secret_annotation` stay public in `coola.display.pydantic`** (they
  are the definition of "secret" shared by `display` and `summary`, ), with their own tests and
  doctests, and they are documented in the guide.
- **One core, two wrappers.** The two public functions keep their names and signatures
  (`str_pydantic_model`, `repr_pydantic_model`); the five options are documented once in
  `str_pydantic_model` and `repr_pydantic_model` points to it ("same options"). Introducing
  `format_pydantic_model(model, *, use_repr=...)` was rejected: the pair mirrors `str`/`repr`, is
  already exported and has the same names as `str_mapping` / `repr_mapping` of `formatting`.

### Terminal output

- **`configure_colorlog_logging(level=logging.INFO, *, force=False, **kwargs)`.** Only the signature
  changes (`force` keyword-only, ). The behaviour, the plain options (`handlers`, `filename`,
  `filemode`, `format`, `style`), the `stream` and `datefmt` handling and the terminal check are
  kept. `Raises` and an `Example` that asserts the handler are added to the docstring.
- **`configure_rich_logging(level=logging.INFO, *, fmt, datefmt, rich_tracebacks, markup, force,
console=None, **kwargs)`.** `console` is new: `RichHandler(console=console or
get_console())`. The docstring loses the text of another project and says that `**kwargs` go to
  `RichHandler` and an unknown name raises `TypeError` (already tested).
- **`make_progressbar(*, transient=False, console=None)` / `make_spinner(*, transient=True,
console=None)`.** Same `console` argument; the columns are unchanged.
- **`print_pretty` / `print_markdown`** keep `title`, `title_align`, `box`, `panel`, `console`.
  `print_markdown(max_length=...)` is **kept and improved**. `max_length` is validated
  (`validate_ge(max_length, 1)`, so a negative value raises `ValueError` instead of cutting from the
  end) and `_truncate` is replaced by a Markdown-aware `_truncate_markdown(msg, max_length)` that
  stays in `print_.py` (it is about Markdown, so it does not belong in `formatting.text`;
  `truncate_str` is not reused because it cuts at a character and appends a suffix). The cut:
  1. snaps to the last paragraph break (`\n\n`), else the last line break, else the last space,
     of the kept text, each only if it keeps at least half of `max_length` (one constant,
     replacing the 80 % rule); otherwise it cuts at `max_length`;
  2. closes a code fence left open by the cut (an odd number of lines that start with ``` or
     `~~~` gets a closing fence), so the rest of the panel is not swallowed by an open block;
  3. appends the notice `*… (N more characters)*` as a separate paragraph, as before (`N` is the
     number of characters of `msg` that are not shown).

  The cut is not guaranteed to keep an inline marker (`**`, `*`, `` ` ``) balanced when it falls
  inside a paragraph: the preference for paragraph and line breaks makes it rare, and it is
  documented instead of repaired with a heuristic. `_print_in_panel` stays (it removes a real duplication) and
  `print_.py` gets its docstring corrected ("Provide helpers to print an object or a Markdown string
  with rich").

- **The `rich` modules stay four files** (`logging`, `print_`, `progressbar`, `__init__`). Merging
  them was rejected: the three cover three different `rich` APIs and the split keeps each import
  list short.

## Public API

```python
from coola.display import (
    BaseDisplayMixin,  # unchanged
    InlineDisplayMixin,  # unchanged
    MultilineDisplayMixin,  # unchanged
    repr_pydantic_model,  # (model, *, sort=True, exclude_none=False, exclude_secret=True, exclude_fields=None)
    str_pydantic_model,  # same options
)
from coola.display.pydantic import is_secret_annotation, secret_field_names  # unchanged
from coola.display.colorlog import (
    configure_colorlog_logging,
)  # (level=INFO, *, force=False, **kwargs)
from coola.display.rich import (
    configure_rich_logging,  # (level=INFO, *, fmt, datefmt, rich_tracebacks, markup, force, console=None, **kwargs)
    make_progressbar,  # (*, transient=False, console=None)
    make_spinner,  # (*, transient=True, console=None)
    print_markdown,  # (msg, *, title, title_align, box, panel, max_length=None, console)
    print_pretty,  # (data, *, title, title_align, box, panel, console)
)
```

The behaviour changes to put in the release notes:

| Before                                                                                                                 | After                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `import coola.display.rich` without `rich`: `ModuleNotFoundError`                                                      | the import works; the first call raises `MissingPackageError` (`pip install coola[rich]`)                  |
| `import coola.display.rich` imports `rich`                                                                             | it imports no `rich` module until a name is used                                                           |
| `configure_colorlog_logging(logging.INFO, True)`                                                                       | `TypeError`: `force` is keyword-only                                                                       |
| `str_pydantic_model(P(c=C(...)))` shows `c={'name': 'a'}`                                                              | shows `c=C(name=a)`; a nested model has its name, with the options of the root                             |
| a field with `Field(repr=False)` is shown; a `field_serializer` runs                                                   | the field is hidden like in pydantic's `repr`; no user code runs                                           |
| a `computed_field` is shown (it is in `model_dump()`)                                                                  | it is not shown (it is not a field); documented                                                            |
| `print_markdown(msg, max_length=-1)` cut from the end silently; an open code fence in the cut text swallows the notice | `ValueError` for `max_length < 1`; the cut prefers paragraph and line breaks and closes an open code fence |
| `configure_rich_logging`, `make_*` always write to `get_console()`                                                     | `console=` chooses the console; default unchanged                                                          |

## Dependents

| Package                                           | Change                                                                                                                                                                  |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `io`, `recursive`, `hashing`, `registry`, `tools` | None: they use `InlineDisplayMixin` / `_display_fields`, unchanged.                                                                                                     |
| `summary.pydantic`                                | None: `secret_field_names` keeps its name and its module.                                                                                                               |
| `coola.utils.fallback`                            | New `rich.py` with the fakes; added to the test of the fallback modules if there is a generic one.                                                                      |
| `coola.testing.marks`                             | None: `rich_available` and `colorlog_available` exist. The `rich` tests use them.                                                                                       |
| `pyproject.toml` (`lint-imports`)                 | None: `display` already is in the core group. A contract "`display` imports `utils`, `validation`, `formatting` only" is added by the `utils` review; no new rule here. |

## Decisions for the maintainers

**D1. A fallback module for `rich`, or an import inside each function?** _Recommended: the fallback
module._ It is the convention of the repository, the names are visible at the top of the module, and
the error is raised by the same mechanism for every library. Alternative: `check_rich()` and
`from rich import ...` inside each of the five functions; no new file, no fake names to keep in sync
with the names imported, but a second convention and imports hidden in function bodies.

Decision: follow recommendation

**D2. Remove `print_markdown(max_length)`?** _Recommended: yes._ It was added in the last release, it
cuts a Markdown string at a character count (it can drop the closing `**` or a code fence, ),
`truncate_str` already exists in `formatting`, and no other helper of the package truncates. Alternative:
keep it, call `truncate_str(msg, max_length, suffix=...)` and validate `max_length >= 1` with
`validate_ge`; this keeps the feature and the mid-syntax cut.

Decision (maintainers): keep it, but improve it. Resolved as: keep `max_length`, validate it, prefer
paragraph and line breaks, close an open code fence, keep the notice; `_truncate` is replaced by
`_truncate_markdown`, and `truncate_str` is not used.

**D3. Should the pydantic formatter show computed fields and run serializers?** _Recommended: no._
The function is documented as a safe formatter for logs, whose value is that it cannot raise or leak
through user code. Alternative: keep `model_dump()` and fix by walking the dump; it keeps
`computed_field` but keeps the parallel walk and the serializers.

Decision: follow recommendation

**D4. Where do `secret_field_names` and `is_secret_annotation` live?** _Recommended: stay in
`coola.display.pydantic`._ `summary` imports them (feature → core is allowed) and nobody else needs
them. Alternative: a `coola.pydantic` helper module for the pydantic knowledge shared by `display`,
`summary`, `hashing` and `equality`; a new top-level package for two functions is not justified
until a third use exists.

Decision: follow recommendation

**D5. One generic `format_pydantic_model(model, *, use_repr)` instead of the pair?** _Recommended: no,
keep the pair._ It mirrors `str`/`repr` and the existing exports; the pair is two wrappers of 5 lines
over the private function. Only the documentation is deduplicated.

Decision: follow recommendation

**D6. Guard the mixins against self-reference?** _Recommended: no._ A `reprlib.recursive_repr`-style
guard needs a per-call set of ids on every `__repr__`, for a case that does not occur in the objects
that use the mixins (configurations, loaders, transformers, registries). Alternative: wrap
`__repr__` with `reprlib.recursive_repr()` (3 lines, returns `...` for the inner call); cheap, but it
changes `__str__` too and hides the bug in the caller's `_display_fields`.

Decision: follow recommendation

**D7. One `configure_logging(*, backend=...)` instead of two functions?** _Recommended: no._ The two
functions have different options (`rich_tracebacks`, `markup` against the colour of the formatter),
and a dispatcher adds a name and a string option for two lines of caller code.

Decision: follow recommendation

## Not covered

- **A `Table` or `Tree` helper** for `rich` (for the summary of `coola.summary`): out of scope;
  `print_pretty` takes any object.
- **Theming of the colours of `colorlog`** (the `log_colors` mapping is fixed): add an option when a
  user asks.
- **Log in JSON or to a file**: `configure_*_logging` forward `handlers`, `filename`, `format` to
  `logging.basicConfig`; nothing else is planned.
- **Moving the logging helpers out of `display`** (to `coola.tools` or a `coola.logging` package):
  `display` is where the user looks for terminal output and the two modules are small; revisit if the
  package grows a third logging backend.
- **`coola.utils.logging.CaptureHandler`** is a test helper for loggers and is unrelated to the
  display of records.
