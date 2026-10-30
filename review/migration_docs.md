# Plan: migrate the documentation from MkDocs to Zensical

Status: proposal. Nothing in this plan has been implemented yet.

## Goal

Build and manage the documentation with [Zensical](https://zensical.org) only. After the
migration the repository has no dependency on `mkdocs`, `mkdocs-material`, `mike` or any
`mkdocs-*` plugin, and no file named `mkdocs.yml`. Breaking changes are accepted when they
improve the contributor or reader experience.

## Current state

| Area                        | Today                                                                                                                   |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Config                      | `docs/mkdocs.yml` (Material theme, nav, markdown extensions, plugins)                                                   |
| Theme override              | `docs/overrides/main.html` (Material `outdated` block of mike)                                                          |
| Plugins                     | `search`, `autorefs`, `mkdocstrings` (python, `griffe-inherited-docstrings`)                                            |
| Dependencies (`docs` group) | `mike`, `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-include-markdown-plugin`, `griffe-inherited-docstrings`      |
| Versioning                  | `mike` publishes to the `gh-pages` branch (`dev` from `main`, `latest` from tags)                                       |
| Publish                     | `inv doc.publish-dev` / `inv doc.publish-latest` (from `invoke-tasklib`, call `mike`)                                   |
| CI                          | `ci-doctest.yaml` (runs `mike deploy` to a temporary branch), `release-docs*.yaml`, composite action `setup-doc-deploy` |
| Local                       | `mkdocs serve -f docs/mkdocs.yml` (`CONTRIBUTING.md`, `docs/docs/dev/development.md`)                                   |
| Strictness                  | `strict: true` and `validation: ... warn` fail the build on broken links                                                |

`mkdocs-include-markdown-plugin` is listed as a dependency but `mkdocs.yml` does not enable it.
Verify with `grep -rn "include-markdown" docs` and drop it if unused.

## What Zensical provides (checked on zensical.org, to re-verify before starting)

- Native config file `zensical.toml`; it can also read `mkdocs.yml`, which is only a bridge.
- `zensical serve` and `zensical build`.
- Supported: `search`, `autorefs`, `mkdocstrings` (plus `mkdocstrings-python`), Material-style
  theme features, and PyMdown extensions.
- No native versioning yet. The only route is a fork of `mike` installed from GitHub
  (not PyPI); Zensical describes it as temporary.
- Unsupported plugins and settings are silently ignored. No documented strict mode.
- `watch` is ignored; sources outside the project directory (`src/`) are not watched by
  `serve`.

The last two points are the risks: silent failures and no strict mode.

## Target state

### 1. One native config: `zensical.toml`

- Move `docs/mkdocs.yml` to `zensical.toml` (at the repository root, which is what `zensical`
  looks for by default), converting site metadata, `nav`, theme features, palette, markdown
  extensions and the mkdocstrings options.
- Delete `docs/mkdocs.yml`. Remove the `!!python/name:material.extensions.emoji...` entries:
  they import `material`, which goes away. Use the Zensical/PyMdown emoji configuration instead.
- Rename the `invoke.yaml` key `docs_config` to point at `zensical.toml`.
- Keep content in `docs/docs/`. Consider a later flattening to `docs/` (drop the double `docs`)
  as a separate PR, because it changes every path in the CI and tasks.

### 2. Dependencies

`docs` group in `pyproject.toml`:

```toml
docs = [
  "griffe-inherited-docstrings >=1.1,<2.0",
  "mkdocstrings-python >=...,<...",
  "zensical >=...,<...",
]
```

- Removed: `mkdocs-material`, `mike`, `mkdocs-include-markdown-plugin`, and the `mkdocstrings[python]`
  extra (it pulls `mkdocs` through `mkdocstrings`; check whether `mkdocstrings` itself still
  requires `mkdocs` and, if it does, record that as a known transitive dependency that cannot
  be removed until upstream changes).
- Pin `zensical` with an upper bound: it is pre-1.0 and its config can still change.

### 3. Strict build (replace `strict: true`)

Zensical has no documented strict mode, so recover the guarantee elsewhere:

1. Run `zensical build` in CI and fail on any warning in its output (`2>&1 | tee` plus a grep
   for `WARNING`/`ERROR`), once the exact log format is known.
2. Add a link check on the built `site/` directory (for example `lychee` for internal links
   and anchors), run in `ci-docs.yaml`.
3. Keep the doctests of the markdown files (`inv test.doctest-markdown`), which already
   cover code blocks.
4. Open an upstream issue/check the Zensical roadmap for a strict flag, then replace 1 with it.

### 4. Versioning and publishing

Options, in order of preference:

1. **Native Zensical versioning**, when released. Until then, do not migrate publishing.
2. **Single version site** (breaking, but simplest): publish only `main` to GitHub Pages with
   `actions/deploy-pages` (no `gh-pages` branch, no git identity, no `mike`). Old versions stay
   reachable through the git tags and the PyPI sdist. Drop the version selector.
3. **Fork of mike** pinned to a commit, as an interim state only.

Recommendation: do the build/serve migration first (PRs 1-3 below), keep publishing on the
mike fork pinned to a commit while Zensical has no versioning, and move to option 1 or 2
afterwards. Decide between 1 and 2 based on whether readers need documentation of old versions.
This decision is the open question for the maintainer.

When publishing no longer uses mike:

- Delete `.github/actions/setup-doc-deploy` and the `CI_BOT_EMAIL` secret use.
- Replace `release-docs-publish.yaml` by a single `docs.yaml`: build the site, upload it with
  `actions/upload-pages-artifact`, deploy with `actions/deploy-pages`.
- Delete `docs/overrides/main.html` (it is the mike "outdated" banner).
- Remove `doc.publish-dev` / `doc.publish-latest` usage; add `doc.build` and `doc.serve` tasks
  that call `zensical` (in `invoke-tasklib` or in `tasks.py`).

### 5. Contributor workflow

- `inv doc.serve` and `inv doc.build` replace the raw `mkdocs ...` commands in
  `CONTRIBUTING.md` and `docs/docs/dev/development.md`.
- `serve` does not watch `src/`: document that docstring edits need a restart, or add a small
  `watchfiles`-based wrapper if that proves painful.
- Update the project layout in `development.md` (`mkdocs.yml` becomes `zensical.toml`) and the
  "MkDocs Documentation" link.
- `.gitignore`: replace the `# mkdocs documentation` / `docs/site` entries with the Zensical
  output directory (check the default; `site/`).
- `codespell` skip list in `pyproject.toml` (`./docs/site`) follows the output directory.

### 6. CI

| Workflow                                                                  | Change                                                                                                                                           |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ci-doctest.yaml`                                                         | Split the docs build into its own job: `zensical build` and the link check. Remove the temporary-branch `mike deploy` and the git identity step. |
| `release-docs.yaml`, `release-docs-dev.yaml`, `release-docs-publish.yaml` | Replace by the Pages workflow of section 4 (or keep, pinned to the mike fork, in the interim).                                                   |
| `ci-verify-workflows.yaml`                                                | Unchanged; it validates the new files.                                                                                                           |

## Migration steps (one PR each, per `CLAUDE.md`)

1. **Spike (no merge)**: `zensical build` on the current `mkdocs.yml`; diff the page list,
   API pages, anchors, search and emoji rendering against the current site. Record the
   differences in this file.
2. **Config**: add `zensical.toml`, delete `mkdocs.yml`, swap dependencies, add `doc.build` /
   `doc.serve`, update `.gitignore`, `CONTRIBUTING.md`, `development.md`.
3. **CI build check**: docs job with warnings-as-errors and the link check.
4. **Publishing**: switch per the decision in section 4; remove `overrides/`,
   `setup-doc-deploy`, and the mike references in workflow comments.
5. **Cleanup**: remove stale mentions (`grep -rniE 'mkdocs|material|mike' . --exclude-dir=.venv
--exclude-dir=.git`), update `review/suggestions.md` with the PR numbers.

## Verification

- `zensical build` exits 0 with no warnings; every page of the old nav is present.
- API pages contain the same objects as before, including inherited docstrings.
- Internal links and anchors pass the link checker.
- Light/dark palette, code copy button, tabs navigation and search work in the browser
  (use the `run` skill or a manual check on `zensical serve`).
- `uv.lock` no longer contains `mkdocs`, `mkdocs-material` or `mike`
  (`grep -n '^name = "\(mkdocs\|mike\)' uv.lock`), or the exception is documented.

## Risks

| Risk                                                        | Mitigation                                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------------- |
| Silent ignore of unsupported options                        | Spike diff in step 1; link check in CI                           |
| No strict mode                                              | Warning grep and link checker (section 3)                        |
| Pre-1.0 tool, config changes                                | Upper-bound pin; Dependabot/`bot-update-deps` PRs build the docs |
| No native versioning                                        | Interim mike fork pinned to a commit, or single-version site     |
| `mkdocstrings` may keep `mkdocs` as a transitive dependency | Check in the spike; document it                                  |
| `serve` ignores `src/` changes                              | Document, or wrapper task                                        |

## Breaking changes (accepted)

- The docs config file and the local commands change (`mkdocs serve` becomes `inv doc.serve`).
- The version selector and the "not the latest version" banner may disappear (option 2).
- Old documentation URLs under `/coola/<version>/` stop being updated; keep a redirect from
  the root to `main`/`latest` if the old site stays up.
