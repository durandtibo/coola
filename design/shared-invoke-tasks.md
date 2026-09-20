# Sharing `invoke` tasks across projects

## Problem

Many projects define nearly identical `invoke` commands in `tasks.py` (lint, test,
docs, release, git helpers, ...). Only small things differ between projects, mainly
the package name and a few paths. We want to reuse these commands the way a shared
Makefile include would be reused, instead of copy-pasting `tasks.py` into every repo.

## Options considered

### Option 1 — Separate pip-installable package

Put the shared tasks in an installable package (e.g. `invoke-tasklib`).
Each project's `tasks.py` imports the tasks it needs and composes its own
`Collection`, with project-specific values passed through `invoke`'s config system.

```python
from invoke import Collection
from invoke_tasklib import lint, test, build

ns = Collection(lint, test, build)
ns.configure({"package": {"name": "coola"}})
```

**Pros**
- Idiomatic Python: real imports, testable task functions, IDE support.
- Versioned and pinned per project — upgrade one repo at a time, roll back if needed.
- Config/code separation: differences live in a small YAML/dict, not duplicated logic.
- Shared tasks can be unit-tested once instead of trusted-by-copy everywhere.

**Cons**
- Extra release ceremony: change a shared task -> bump/publish package -> bump pin
  in each consumer. Slower than editing a file in place.
- Requires a distribution mechanism (private index, or git dependency).
- Adds a layer of indirection when debugging "which version is actually running".
- Overhead of standing up a new package/repo for what is currently just files.

### Option 2 — Git submodule / subtree of a `tasks/` directory

Vendor a shared `tasks/` directory into each repo via submodule or subtree, imported
by the local `tasks.py`. Closest to a literal Makefile include (copied files,
versioned via git), but submodules are high-friction and drift easily. Not pursued
further.

### Option 3 — Config-only variance, identical `tasks.py` everywhere

A narrower variant of Option 1: every project's `tasks.py` is identical, and all
customization happens through `invoke`'s built-in config loading (`invoke.yaml`,
env vars, CLI flags) rather than per-project Python composition.

```python
# tasks.py — identical in every project
from invoke_tasklib import ns
```

```yaml
# invoke.yaml — the only thing that differs per project
package:
  name: coola
paths:
  src: src/coola
  tests: tests
```

**Pros**
- Maximum consistency — literally the same `tasks.py` everywhere.
- Non-Python-comfortable contributors can adjust behavior via YAML only.
- New shared tasks require zero per-project changes beyond bumping the package.
- Uses `invoke`'s native config layering — no custom machinery needed.

**Cons**
- Less flexible: a project needing a genuinely different task set or a one-off task
  either bloats the shared package with conditionals, or falls back to per-project
  code anyway.
- Customization is limited to axes anticipated in advance as config values.
- Debugging becomes "which config layer produced this value" across merged
  YAML/env sources.
- Still carries all the packaging/versioning overhead of Option 1.

## Chosen approach: Hybrid

Default to Option 3 (identical `tasks.py`, config-driven) for the common case, but
allow a project to drop down to Option 1's explicit `Collection` composition when it
truly needs a custom task or subset.

## Package name

- **Name**: `invoke-tasklib` (available on PyPI at the time of writing)
- **Description**: "Reusable Invoke tasks shared across Python projects."

## Plan

### Phase 1 — Extract and design the shared package
1. Create `invoke-tasklib` repo (standalone, or inside an existing internal
   tools monorepo).
2. Inventory current tasks across projects (starting with `coola`'s `tasks.py`) and
   group into modules by concern: `lint.py`, `format.py`, `test.py`, `docs.py`,
   `release.py`, `git.py`, etc.
3. Parameterize hardcoded values (package name, src/test paths, doc source dir,
   publish target) by reading from `ctx.config.*`, with sane defaults defined in the
   shared package.
4. Assemble the default namespace: a top-level `ns = Collection(...)` registering
   every shared task, plus `ns.configure(DEFAULTS)`.
5. Add tests for the shared tasks themselves (at least smoke/dry-run tests), since a
   break here now affects every project.
6. Set up packaging + versioning: `pyproject.toml`, semantic version, changelog.
   Decide distribution method (private PyPI index, or `git+https://...@vX.Y.Z` pin
   to start).

### Phase 2 — Define the config contract
1. Document the config schema the shared tasks expect (`package.name`,
   `paths.src`, `paths.tests`, `docs.source`, etc.) in the shared package's README.
2. Standardize on `invoke.yaml` at each project root (auto-loaded by `invoke`).
3. Make every value optional with a default, so a project's `invoke.yaml` only
   needs to state real differences (mainly package name and nonstandard paths).

### Phase 3 — Standard integration path (identical `tasks.py`)
1. Standard per-project `tasks.py`:
   ```python
   from invoke_tasklib import ns
   ```
2. Standard per-project `invoke.yaml`:
   ```yaml
   package:
     name: <project-package-name>
   ```
3. Add the shared package as a pinned dev dependency.

### Phase 4 — Escape hatch for project-specific needs
1. Document the pattern for diverging: import individual task modules instead of
   the pre-built `ns`, and build a custom `Collection`:
   ```python
   from invoke import Collection
   from invoke_tasklib import lint, test, docs
   from . import my_custom_task

   ns = Collection(lint, test, docs, my_custom_task)
   ns.configure({...})
   ```
2. Norm: prefer adding a config knob to the shared task over forking it; reserve
   custom composition for things that are truly one-off.
3. For shared tasks needing a per-project hook (e.g. an extra step before publish),
   consider a documented extension point (optional pre/post callable in config)
   rather than encouraging full overrides.

### Phase 5 — Migrate projects
1. Pilot with `coola`.
2. Migrate fully, verify all tasks behave identically (`invoke --list`, run each
   task).
3. Write a migration guide in the shared package's README: install, `tasks.py`
   template, `invoke.yaml` template, how to bump the shared package version later.
4. Roll out to remaining projects one at a time, keeping the old `tasks.py`
   available (branch/stash) until verified.

### Phase 6 — Maintenance workflow
1. Changes to shared tasks -> bump version in `invoke-tasklib` -> update
   changelog.
2. Each project upgrades its pin independently; CI runs `invoke --list` + key tasks
   after bumping.
3. Periodically audit projects for drift back into full custom `Collection`s that
   could instead be config additions upstream.

## Open decisions

- Where does the shared package live (new standalone repo vs. existing tools repo)?
- Distribution: private index now, or git-pinned dependency to start?
- Confirm `coola` as the pilot migration.
