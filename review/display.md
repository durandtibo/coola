# `coola.display`: target state and the changes to reach it

This document describes the design that `coola.display` should have, then the changes to make to
the current implementation (`main`, `b6161ad4`) to get there. The design follows the rules of the
user guide (`docs/docs/uguide/display.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: lazy imports of the optional libraries, the layering of the core
packages), and it stays aligned with `formatting`, `utils` and the packages that use the mixins
(`io`, `recursive`, `hashing`, `registry`). Its aim is a small package that is easy to maintain:
**three kinds of content (mixins, pydantic formatting, terminal output), one rule for the optional
libraries, one behaviour per option**. Breaking changes are accepted, and the future state is more
important than the past one.

Evidence comes from probes run on `main` (Python 3.14, `.venv`, scripts kept outside the
repository). Items marked **[verified]** were reproduced; **[reading]** comes from the code. Nothing
in `src`, `tests` or `docs` was changed.

## Constraints

1. **Simple and easy to maintain.** The package is 981 lines of `src`, most of them docstrings.
   Remove what duplicates another package (`_truncate`), what walks two structures in parallel
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

## 1. Properties of the sub-package

These are the rules of the user guide, completed. Every design choice below exists to make one of
them true **by construction** instead of by the discipline of each caller.

| Property             | Statement                                                                                                                                                                                                                                                              | Enforced by                                   |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| One hook             | A class that uses a mixin only overrides `_display_fields`. `__repr__` and `__str__` come from the mixin; a class with no field overrides nothing.                                                                                                                     | `BaseDisplayMixin`                            |
| Consistent format    | All the objects that use a mixin share the same layout: `Name(key=value)` inline, or one `(key): value` per line. `__repr__` shows the `repr` of the values and `__str__` their `str`. The name is the `__qualname__` of the class.                                    | `InlineDisplayMixin`, `MultilineDisplayMixin` |
| Pure                 | Formatting never modifies the object and has no side effect (no logging, no I/O, no global state). Two calls on the same object give the same string.                                                                                                                  | the formatting functions                      |
| Secrets never leak   | A pydantic field typed as a secret (`SecretStr`, `SecretBytes`, `Secret`, or a type that contains one) is omitted by default, at any depth (nested models, lists, tuples, dicts). With `exclude_secret=False` the masked value of pydantic is shown, never the secret. | `coola.display.pydantic`                      |
| Honest options       | An option that names a field that does not exist (`exclude_fields`) warns, at the line of the caller. An option out of its range raises `ValueError`.                                                                                                                  | `coola.validation`, `warnings.warn`           |
| Optional libraries   | `pydantic`, `rich` and `colorlog` are imported when they are used, never when `coola` or `coola.display` is imported. A missing library raises `MissingPackageError` (what to install) when a function that needs it is called, and never at import time.              | `lazy_exports`, `coola.utils.fallback`        |
| Safe terminal output | Colour is used only if the library is installed **and** the output is a terminal (`colorlog`), or the library decides (`rich` strips the styles of a file). Otherwise the output is plain: no ANSI code in a file or in CI.                                            | `configure_colorlog_logging`, `rich`          |
| Respects the logger  | `logging.basicConfig` does nothing if the root logger already has handlers, unless `force=True`. Both `configure_*_logging` functions have the same options for this: `level`, `force`, `**kwargs` forwarded.                                                          | `logging.basicConfig`                         |
| One console          | The `rich` helpers (`print_*`, progress bars, logging) write to the one console of `rich` (`rich.get_console()`) unless a `console` is passed, so their output does not interleave.                                                                                    | the `rich` helpers                            |
| Keyword-only options | Every option after the first required arguments (and `level` for the logging functions) is keyword-only.                                                                                                                                                               | the signatures                                |

Not guaranteed, and documented as such:

- **Self-referencing objects.** `repr` of an object whose `_display_fields` contains itself raises
  `RecursionError` like any `__repr__` that recurses (**[verified]**). The mixin does not guard it:
  the standard library does not either for a plain class, and a guard needs per-call state on a
  hot path (decision D6).
- **The layout of a multiline value.** A value whose `str` has several lines is shown with its
  lines indented under the key; this is a readability rule, not a format to parse.
- **The look of the `rich` output** (box characters, colours, widths) depends on the terminal and on
  the version of `rich`; only the content is tested.

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Status         |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **`coola.display.rich` breaks without `rich`.** `__init__.py` imports `logging.py`, `print_.py` and `progressbar.py`, and each imports `rich` at module level with no guard. `import coola.display.rich` raises `ModuleNotFoundError: import of rich halted` (no hint to install `coola[rich]`) and, when `rich` is installed, importing the package imports `rich`, which the architecture guide says a subpackage must not do. `check_rich` exists in `coola.utils.imports` and is used nowhere in `src`. | **[verified]** |
| F2  | **The `rich` tests fail at collection without `rich`.** `tests/unit/display/rich/*.py` import `rich` (or `coola.display.rich`) at the top of the module and use no `rich_available` mark, so a CI job without `rich` errors instead of skipping. `coola.testing.marks` already provides the mark.                                                                                                                                                                                                           | **[reading]**  |
| F3  | **There is no fallback module for `rich`**, unlike `colorlog`, `pydantic`, `numpy`, ... (`coola.utils.fallback`). `colorlog.py` and `pydantic.py` of this package follow the pattern; the three `rich` modules do not, so the same package has two conventions.                                                                                                                                                                                                                                             | **[reading]**  |
| F4  | **`_truncate` duplicates `coola.formatting.text.truncate_str`** and does it worse: `_truncate("abcdef", -1)` returns `("abcde", True)` (a negative `max_length` silently cuts from the end), and a Markdown string cut in the middle of `**...**` leaves an unbalanced marker, so the output shows a literal `**word word wor` **[verified]**. A cut of Markdown text is not a text truncation: it changes the meaning of the syntax.                                                                       | **[verified]** |
| F5  | **Text from another project in the docstrings.** `configure_rich_logging` says `langchain_metrics.console`; `make_progressbar` says "across the codebase", "Processing papers"; the module docstring of `print_.py` says "Provide a shared Rich console" (it defines none). `uguide/display.md` and `refs/display.md` do not mention `coola.display.rich` at all (and `refs` omits `coola.display.pydantic`).                                                                                               | **[reading]**  |
| F6  | **The two logging functions disagree.** `configure_colorlog_logging(level, force=False, **kwargs)` takes `force` positionally; `configure_rich_logging(level, *, fmt, datefmt, ..., force, **kwargs)` is keyword-only. `CLAUDE.md`: new options are keyword-only. The colorlog docstring also has no `Raises` and no `Example` with an observable result.                                                                                                                                                   | **[reading]**  |
| F7  | **The pydantic formatting loses the type of the nested models.** `model_dump()` turns every nested model into a `dict`, so `str_pydantic_model(P(c=C(name="a", t="x")))` is `P(c={'name': 'a'}, ...)` where `str(P(...))` shows `c=C(name='a', ...)` **[verified]**. Fields declared with `Field(repr=False)` are shown (pydantic's own `repr` hides them), and a `field_serializer` (user code) runs inside a function that is meant to be a safe, pure formatter.                                         | **[verified]** |
| F8  | **The secret masking walks two structures in parallel.** `_mask_secret_fields` / `_mask_nested_secret_fields` (45 lines, 3 `cast`) zip the model with its dump and rebuild lists, tuples and dicts; `zip(..., strict=False)` and `if k in dumped_map` hide a length mismatch instead of reporting it. It exists only because the dump was taken first (F7).                                                                                                                                                 | **[reading]**  |
| F9  | **`str_pydantic_model` and `repr_pydantic_model` repeat 5 options and 25 lines of docstring**, differ only by `mapping_line_fn`, and are not consistent with the mixin API (`use_repr`), which already is the shared knob.                                                                                                                                                                                                                                                                                  | **[reading]**  |
| F10 | **`is_secret_annotation` and `secret_field_names` are public, used by `summary`, and tested only through the formatters.** `coola.summary.pydantic` imports them from `coola.display.pydantic`: a feature package depends on a helper of a display module. It works (direction feature → core) but the helper is not about display.                                                                                                                                                                         | **[reading]**  |
| F11 | **Hidden mutable behaviour of the Rich `console` argument.** `_print_in_panel` accepts `console or get_console()`; `configure_rich_logging` and the progress helpers always use `get_console()` and have no `console` argument, so a program that renders to its own `Console` (tests, a file) cannot log or show progress there.                                                                                                                                                                           | **[reading]**  |
| F12 | **Guide Properties are only half true.** "Safe terminal output" and "Respects the logger" are about two functions of one module; the mixins have no property for `repr`/`str` of nested values, the pydantic helpers have none at all, and there is no statement about the optional libraries.                                                                                                                                                                                                              | **[reading]**  |

What is already right and must not regress: the three mixins and the `_display_fields` hook (used by
`io`, `recursive`, `hashing`, `registry`, `tools.timing`); the layout of both mixins; the lazy export
of the pydantic functions from `coola.display`; the terminal check and the `stream` handling of
`configure_colorlog_logging` (`sys.stderr is None`, a stream without `isatty`); the warning
`stacklevel` of `exclude_fields` (points at the caller, **[verified]**); the keyword-only options of
`configure_rich_logging`, `print_*` and `make_*`; 100 % statement and branch coverage of the
package.

## 3. Architecture

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

### 3.1 The rule for an optional library (new for `rich`)

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
`builtins.__import__`. The cost is one new file of 25 lines (decision D1).

### 3.2 The mixins (unchanged)

`BaseDisplayMixin` defines `__repr__`, `__str__` and a default `_display_fields`; the two concrete
mixins implement `_render_fields`. Nothing changes except the guide. One remark kept for the
maintainers: `MultilineDisplayMixin` uses `repr_indent(repr_mapping(fields))`, which indents the
continuation lines of a multiline value (`(y): a\n  b`). This is intended and covered by a test.

### 3.3 Formatting a pydantic model (simplified)

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

- **Nested models keep their name** (`P(c=C(name=a), items=[C(name=b)])`, F7). A nested model is
  formatted with the same options as the root (`sort`, `exclude_none`, `exclude_secret`); the
  shown value is a tiny object whose `__repr__`/`__str__` is the formatted string, so the generic
  `repr_mapping_line` and `str_mapping_line` of `coola.formatting` need no change.
- **No parallel walk, no `cast`** (F8): the value is read once, from the model; a nested secret is
  dropped by the recursive call on the nested model, so the 45 lines of `_mask_*` and the
  `zip(strict=False)` disappear (about 20 lines for `_shown`).
- **No serializer runs.** `getattr` reads the attribute; a `field_serializer`, an alias or a
  `computed_field` has no effect (a `computed_field` is not a field, so it is not shown; pydantic's
  `repr` shows it, the function does not: it is a property that may be expensive). This is stated
  in the docstring.
- **`secret_field_names` and `is_secret_annotation` stay public in `coola.display.pydantic`** (they
  are the definition of "secret" shared by `display` and `summary`, F10), with their own tests and
  doctests, and they are documented in the guide (see D4).
- **One core, two wrappers.** The two public functions keep their names and signatures
  (`str_pydantic_model`, `repr_pydantic_model`); the five options are documented once in
  `str_pydantic_model` and `repr_pydantic_model` points to it ("same options"). Introducing
  `format_pydantic_model(model, *, use_repr=...)` was rejected: the pair mirrors `str`/`repr`, is
  already exported and has the same names as `str_mapping` / `repr_mapping` of `formatting` (D5).

### 3.4 Terminal output

- **`configure_colorlog_logging(level=logging.INFO, *, force=False, **kwargs)`.** Only the signature
  changes (`force` keyword-only, F6). The behaviour, the plain options (`handlers`, `filename`,
  `filemode`, `format`, `style`), the `stream` and `datefmt` handling and the terminal check are
  kept. `Raises` and an `Example` that asserts the handler are added to the docstring.
- **`configure_rich_logging(level=logging.INFO, *, fmt, datefmt, rich_tracebacks, markup, force,
console=None, **kwargs)`.** `console` is new (F11): `RichHandler(console=console or
get_console())`. The docstring loses the text of another project (F5) and says that `**kwargs` go to
  `RichHandler` and an unknown name raises `TypeError` (already tested).
- **`make_progressbar(*, transient=False, console=None)` / `make_spinner(*, transient=True,
console=None)`.** Same `console` argument; the columns are unchanged.
- **`print_pretty` / `print_markdown`** keep `title`, `title_align`, `box`, `panel`, `console`.
  `print_markdown(max_length=...)` is **removed** (F4, D2): the helper cuts Markdown mid-syntax and
  duplicates `truncate_str`; a caller who wants a short text calls `truncate_str` before. `_truncate`
  and its 7 tests are deleted. `_print_in_panel` stays (it removes a real duplication) and
  `print_.py` gets its docstring corrected ("Provide helpers to print an object or a Markdown string
  with rich").
- **The `rich` modules stay four files** (`logging`, `print_`, `progressbar`, `__init__`). Merging
  them was rejected: the three cover three different `rich` APIs and the split keeps each import
  list short.

## 4. Public API after the change

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
    print_markdown,  # (msg, *, title, title_align, box, panel, console)      no max_length
    print_pretty,  # (data, *, title, title_align, box, panel, console)
)
```

The behaviour changes to put in the release notes:

| Before                                                               | After                                                                                     |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `import coola.display.rich` without `rich`: `ModuleNotFoundError`    | the import works; the first call raises `MissingPackageError` (`pip install coola[rich]`) |
| `import coola.display.rich` imports `rich`                           | it imports no `rich` module until a name is used                                          |
| `configure_colorlog_logging(logging.INFO, True)`                     | `TypeError`: `force` is keyword-only                                                      |
| `str_pydantic_model(P(c=C(...)))` shows `c={'name': 'a'}`            | shows `c=C(name=a)`; a nested model has its name, with the options of the root            |
| a field with `Field(repr=False)` is shown; a `field_serializer` runs | the field is hidden like in pydantic's `repr`; no user code runs                          |
| a `computed_field` is shown (it is in `model_dump()`)                | it is not shown (it is not a field); documented                                           |
| `print_markdown(msg, max_length=n)`                                  | `TypeError`: truncate with `coola.formatting.text.truncate_str`                           |
| `configure_rich_logging`, `make_*` always write to `get_console()`   | `console=` chooses the console; default unchanged                                         |

## 5. Dependents

| Package                                           | Change                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `io`, `recursive`, `hashing`, `registry`, `tools` | None: they use `InlineDisplayMixin` / `_display_fields`, unchanged.                                                                                                                                                                                                                                                                |
| `summary.pydantic`                                | None: `secret_field_names` keeps its name and its module.                                                                                                                                                                                                                                                                          |
| `coola.utils.fallback`                            | New `rich.py` with the fakes; added to the test of the fallback modules if there is a generic one.                                                                                                                                                                                                                                 |
| `coola.testing.marks`                             | None: `rich_available` and `colorlog_available` exist. The `rich` tests use them.                                                                                                                                                                                                                                                  |
| `docs`                                            | `uguide/display.md`: new Properties, a section on `coola.display.rich`, the pydantic options; `refs/display.md`: add `coola.display.pydantic` and `coola.display.rich`; `dev/architecture.md`: add `rich` and `colorlog` to **Optional Dependencies** (today the list stops at `pyarrow`); migration page: the table of section 4. |
| `pyproject.toml` (`lint-imports`)                 | None: `display` already is in the core group. A contract "`display` imports `utils`, `validation`, `formatting` only" is added by the `utils` review (step 3 there); no new rule here.                                                                                                                                             |

## 6. Tests and gates

The conventions of `CLAUDE.md` apply: one behaviour per test, literal expected values, a regression
test for every bug (checked to fail with `git stash push src`), no local imports, `*_available`
marks for the optional libraries.

- **Optional libraries (integration, `tests/integration/display/test_optional_imports.py`, new).**
  A subprocess (as `loaded_libraries` in `test_lazy_imports.py`) per case, no patching:
  - `import coola.display; import coola.display.rich` imports neither `rich`, `pydantic` nor
    `colorlog` (`sys.modules` after the import);
  - with `sys.modules["rich"] = None`, `import coola.display.rich` succeeds and each of the five
    functions raises `MissingPackageError` whose message contains `pip install coola[rich]`
    (parametrised);
  - with `sys.modules["rich"] = None`, `configure_rich_logging()` leaves the handlers of the root
    logger unchanged;
  - with `sys.modules["colorlog"] = None`, `configure_colorlog_logging(force=True)` configures plain
    logging (kept from the unit test, now with a real missing library).
- **Fallback** (`tests/unit/utils/fallback/test_rich.py`): the fakes raise the error with the
  package name (one parametrised test over the names), `MINIMAL` and `ROUNDED` are `None`.
- **Rich unit tests** (`tests/unit/display/rich/`): add `rich_available` to every test module (and
  a guarded import at the top, as `tests/unit/display/test_pydantic.py` does); the modules are
  skipped, not errors, without `rich`. New tests: `console=` is used by `configure_rich_logging`,
  `make_progressbar` and `make_spinner` (`progress.console is console`); `print_markdown` without
  `max_length`; the removal of `_truncate`'s 7 tests.
- **Pydantic** (`test_pydantic.py`): new tests for a nested model that keeps its name, for the options of
  the root applied to a nested model (`sort`, `exclude_none`, `exclude_secret`), for a nested model
  in a list, a tuple, a set and a dict value, for `Field(repr=False)`, for `extra="allow"` values, for a
  `field_serializer` that raises (the formatter does not call it), and the **whole output** of a
  model with nested secrets asserted against a literal. `_mask_*` tests are replaced by tests of
  `_shown`.
- **Mixins** (`test_mixin.py`): add a test of the `RecursionError` documentation (a self-referencing
  field), and one with a value whose `str` has several lines (`MultilineDisplayMixin`).
- **Colorlog** (`test_colorlog.py`): `force` positional raises `TypeError`; the rest is kept.
- **Doctests** (`xdoctest`): the examples of the `rich` functions print to the console and have no
  checked output. Keep them as smoke tests, and add one example per function with a `Console`
  that writes to a `StringIO` and an asserted result (`console.file.getvalue()`), so the guide
  shows a result and not only a call.
- **Coverage**: 100 % of `src/coola/display` and of `src/coola/utils/fallback/rich.py`
  (`--cov=coola`), no `# pragma: no cover` except the two guarded import blocks, as in the rest of
  the package.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                              | Breaking     |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| 1    | **Make `coola.display.rich` optional (F1, F2, F3).** Add `coola/utils/fallback/rich.py`, guard the imports of the three modules, make `rich/__init__.py` lazy with `lazy_exports`, add the `rich_available` marks to the tests, add the integration tests of section 6. No change of behaviour when `rich` is installed.            | no           |
| 2    | **Clean the `rich` helpers (F4, F5, F11).** Remove `print_markdown(max_length)` and `_truncate`, add the `console` argument, remove the text of another project, add the examples with a `StringIO` console.                                                                                                                        | yes          |
| 3    | **Align the logging functions (F6).** `configure_colorlog_logging(level, *, force, **kwargs)`; docstrings (`Raises`, `Example`, the difference between `rich` (required) and `colorlog` (optional)).                                                                                                                                | yes          |
| 4    | **Simplify the pydantic formatting (F7, F8, F9).** Read the fields from the model, `_shown` for the nested values, delete `_mask_*`, one documented option list. New tests, regression tests for F7 checked with `git stash push src`.                                                                                              | yes (output) |
| 5    | **Documentation (F5, F12).** Rewrite `uguide/display.md` (the Properties of section 1 at the top, one section per kind of content, the optional libraries), `refs/display.md`, the optional dependencies and the layering paragraphs of `dev/architecture.md`, the migration page, and the status table of `review/suggestions.md`. | no           |

Steps 1 and 2 are independent of steps 3 and 4; step 5 comes last (or is split in each PR when the PR
changes the page it touches).

## 8. Decisions for the maintainers

**D1. A fallback module for `rich`, or an import inside each function?** _Recommended: the fallback
module._ It is the convention of the repository, the names are visible at the top of the module, and
the error is raised by the same mechanism for every library. Alternative: `check_rich()` and
`from rich import ...` inside each of the five functions; no new file, no fake names to keep in sync
with the names imported, but a second convention and imports hidden in function bodies.

Decision: follow recommendation

**D2. Remove `print_markdown(max_length)`?** _Recommended: yes._ It was added in the last release, it
cuts a Markdown string at a character count (it can drop the closing `**` or a code fence, F4),
`truncate_str` already exists in `formatting`, and no other helper of the package truncates. Alternative:
keep it, call `truncate_str(msg, max_length, suffix=...)` and validate `max_length >= 1` with
`validate_ge`; this keeps the feature and the mid-syntax cut.

Decision: keep it, but improve it

**D3. Should the pydantic formatter show computed fields and run serializers?** _Recommended: no._
The function is documented as a safe formatter for logs, whose value is that it cannot raise or leak
through user code. Alternative: keep `model_dump()` and fix F7 by walking the dump; it keeps
`computed_field` but keeps the parallel walk (F8) and the serializers.

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

## 9. Not covered

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

## Appendix: effect on the size of the package

| Module                    | Lines today | Expected after | Why                                                                               |
| ------------------------- | ----------- | -------------- | --------------------------------------------------------------------------------- |
| `pydantic.py`             | 262         | about 215      | `_mask_*` (45 lines) replaced by `_shown` (about 20), the options documented once |
| `rich/print_.py`          | 202         | about 175      | no `_truncate`, no `max_length`                                                   |
| `rich/logging.py`         | 106         | about 105      | guarded import, `console`, no text of another project                             |
| `rich/progressbar.py`     | 113         | about 120      | guarded import, `console`                                                         |
| `rich/__init__.py`        | 15          | about 25       | `lazy_exports`                                                                    |
| `colorlog.py`             | 111         | about 115      | `*`, `Raises`, `Example`                                                          |
| `utils/fallback/rich.py`  | 0           | about 25       | new                                                                               |
| `mixin.py`, `__init__.py` | 172         | 172            | unchanged                                                                         |

The line counts are estimates from the code to delete and to add; the gate is the behaviour
(sections 1 and 6), not the size.
