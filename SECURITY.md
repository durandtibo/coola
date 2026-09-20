# Security Policy

## Supported Versions

Security fixes are only released for the latest minor version on PyPI. Older versions are not
patched, so please upgrade to the latest release before reporting an issue.

## Reporting a Vulnerability

The `coola` team takes security bugs seriously. We appreciate your efforts to responsibly disclose
your findings.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities privately using
[GitHub Security Advisories](https://github.com/durandtibo/coola/security/advisories/new).

### What to Expect

- We will acknowledge receipt of your report within **5 business days**.
- We will provide an initial assessment (including whether the report is accepted, and an
  estimated timeline for a fix) within **10 business days** of acknowledgment.

### What to Include

Please include the following information in your report:

- Type of issue (e.g., denial of service via crafted input, arbitrary code execution, information
  disclosure, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

### Security Update Policy

- Security updates will be released as soon as possible
- Security updates will be clearly marked in release notes
- We will notify users through GitHub releases and other appropriate channels

## Known Security Considerations

`coola` provides utilities to compare, summarize, and format Python objects (including NumPy,
PyTorch, pandas, and other optional dependencies). Most of the library does not perform network
I/O or execute arbitrary code, with one notable exception:

- **`coola.factory` executes arbitrary code by design**: `import_object`, `instantiate_object`,
  `factory`, and `resolve_object` import a module and object (e.g. `"os.system"`) from a string
  path and, for `instantiate_object`/`factory`, call it with the given arguments. Importing a
  module runs its top-level code, and the resolved object can be anything, including a callable
  that executes shell commands. **Never pass an object path, or a configuration dict containing
  one (e.g. under the `_target_` key), that comes from untrusted input** — a user-uploaded
  config file, a network payload, or any other data you do not fully trust. Treat these functions
  the same way you would treat `eval`/`pickle.load` on untrusted data.
- **`coola.io` pickle loading executes arbitrary code by design**: `PickleLoader.load` and
  `load_pickle` call Python's `pickle.load` on the given file. Unpickling can run arbitrary code
  as a side effect of deserialization regardless of how the pickle was produced. **Only load
  pickle files from a source you trust**; never load one that came from an untrusted or
  unauthenticated source (e.g. a file uploaded by a third party or fetched over the network).
- **Resource exhaustion**: Comparing or summarizing very large or deeply nested/recursive data
  structures can consume significant memory, CPU, or stack depth (recursion). Do not run
  comparisons on untrusted, unbounded input without limits.
- **Sensitive data in output**: Comparison failures and summaries may include the values being
  compared. Avoid logging or displaying `coola` output for objects that contain secrets or
  personal data.
- **Custom extensions**: If you register custom comparators/handlers for your own types, apply the
  same input validation you would to any other code path that processes untrusted data.
