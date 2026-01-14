# Contributing to `coola`

We want to make contributing to this project as easy and transparent as possible.

## Overview

We welcome contributions from anyone, even if you are new to open source.

- If you are planning to contribute back bug-fixes, please do so without any further discussion.
- If you plan to contribute new features, utility functions, or extensions to the core, please first
  open an issue and discuss the feature with us.

Once you implement and test your feature or bug-fix, please submit a Pull Request.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- Git

### Setting Up Your Development Environment

1. **Fork and clone the repository:**
   ```shell
   git clone https://github.com/YOUR-USERNAME/coola.git
   cd coola
   ```

2. **Install Poetry** (if not already installed):
   ```shell
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Create a virtual environment** (optional but recommended):
   ```shell
   make conda
   conda activate coola
   ```

   Or using venv:
   ```shell
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```shell
   poetry install --no-interaction
   ```

5. **Install pre-commit hooks:**
   ```shell
   poetry run pre-commit install
   ```

## Development Workflow

### Running Tests

Run all unit tests:
```shell
make unit-test
```

Run tests with coverage:
```shell
make unit-test-cov
```

Run specific tests:
```shell
poetry run pytest tests/unit/path/to/test_file.py
```

### Code Quality

**Format your code:**
```shell
poetry run black .
```

**Run linter:**
```shell
make lint
```

**Format docstrings:**
```shell
make docformat
```

**Run all pre-commit checks:**
```shell
poetry run pre-commit run --all-files
```

### Building Documentation

Build documentation locally:
```shell
poetry run mkdocs serve -f docs/mkdocs.yml
```

Then visit http://127.0.0.1:8000 to view the documentation.

## Pull Requests

We actively welcome your pull requests.

### Pull Request Process

1. **Fork the repo and create your branch from `main`:**
   ```shell
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes:**
   - Add your code
   - Write tests for new functionality
   - Update documentation if needed

3. **Ensure tests pass:**
   ```shell
   make unit-test-cov
   ```

4. **Run code quality checks:**
   ```shell
   poetry run pre-commit run --all-files
   ```

5. **Commit your changes:**
   ```shell
   git add .
   git commit -m "Add feature: description of feature"
   ```

6. **Push to your fork:**
   ```shell
   git push origin feature/my-new-feature
   ```

7. **Submit a Pull Request** through GitHub

### Pull Request Guidelines

- **Write clear commit messages:** Follow the format "Add/Fix/Update: description"
- **Keep changes focused:** One feature or bug fix per PR
- **Add tests:** All new code should have tests
- **Update documentation:** If you change APIs or add features
- **Follow code style:** Run pre-commit hooks before committing
- **Link related issues:** Reference any related issues in your PR description

### What to Include in Your PR

- **Description:** Clear explanation of what you changed and why
- **Tests:** Unit tests that cover your changes
- **Documentation:** Updates to docs if you changed behavior

## Issues

We use GitHub issues to track public bugs or feature requests.

### Reporting Bugs

When reporting bugs, please include:

1. **Clear title:** Brief description of the issue
2. **Environment information:**
   - `coola` version: `pip show coola`
   - Python version: `python --version`
   - Operating system
3. **Steps to reproduce:** Minimal code example
4. **Expected behavior:** What you expected to happen
5. **Actual behavior:** What actually happened
6. **Error messages:** Full error traceback if applicable

**Example bug report:**

```markdown
## Bug: objects_are_equal fails with custom type

**Environment:**
- coola version: 0.9.1
- Python version: 3.10.8
- OS: Ubuntu 22.04

**Code to reproduce:**
\`\`\`python
from coola.equality import objects_are_equal

class MyClass:
    def __init__(self, value):
        self.value = value

obj1 = MyClass(42)
obj2 = MyClass(42)
objects_are_equal(obj1, obj2)
\`\`\`

**Expected:** Returns True

**Actual:** Raises TypeError: ...

**Error:**
\`\`\`
Traceback (most recent call last):
  ...
\`\`\`
```

### Requesting Features

For feature requests, please include:

1. **Clear title:** Brief description of the feature
2. **Motivation:** Why this feature would be useful
3. **Proposed solution:** How you envision it working
4. **Alternatives considered:** Other approaches you've thought about
5. **Examples:** Code examples showing how it would be used

## Coding Standards

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Maximum line length: 100 characters

### Docstring Style

- Follow [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints in function signatures
- Document all public APIs

**Example:**

```python
def my_function(param1: int, param2: str) -> bool:
    """Brief description of the function.

    Longer description if needed. Explain what the function does,
    any important notes, etc.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.

    Example:
        ```python
        result = my_function(42, "hello")
        print(result)  # True
        ```
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return param2.startswith("h")
```

### Testing Standards

- Write unit tests for all new code
- Aim for >90% code coverage
- Use descriptive test names: `test_<function>_<condition>_<expected_result>`
- Use fixtures for common test data
- Test edge cases and error conditions

**Example test:**

```python
import pytest
from coola.equality import objects_are_equal


def test_objects_are_equal_with_identical_dicts_returns_true():
    """Test that objects_are_equal returns True for identical dictionaries."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"a": 1, "b": 2}
    assert objects_are_equal(dict1, dict2)


def test_objects_are_equal_with_different_dicts_returns_false():
    """Test that objects_are_equal returns False for different dictionaries."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"a": 1, "b": 3}
    assert not objects_are_equal(dict1, dict2)
```

## Documentation

### Documentation Standards

- Update documentation for all user-facing changes
- Include code examples in documentation
- Keep documentation up-to-date with code changes
- Use clear, concise language

### Documentation Types

1. **API documentation:** Auto-generated from docstrings
2. **User guides:** Step-by-step tutorials
3. **Examples:** Practical code examples
4. **FAQ:** Common questions and answers

## Commit Message Guidelines

Follow these conventions:

- **Add:** New feature or functionality
- **Fix:** Bug fix
- **Update:** Changes to existing functionality
- **Remove:** Removal of code or features
- **Refactor:** Code changes that don't fix bugs or add features
- **Docs:** Documentation changes
- **Test:** Adding or updating tests
- **Build:** Changes to build system or dependencies

**Examples:**
- `Add: Support for custom comparators`
- `Fix: Handle NaN values correctly in numpy arrays`
- `Update: Improve error messages for type mismatches`

## Community

### Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

### Getting Help

- **Documentation:** https://durandtibo.github.io/coola/
- **Issues:** https://github.com/durandtibo/coola/issues
- **Discussions:** Open an issue for questions

### Recognition

Contributors will be:
- Listed in release notes
- Credited in the contributors list
- Mentioned in relevant documentation

## License

By contributing to `coola`, you agree that your contributions will be licensed under the
BSD 3-Clause License as specified in the [LICENSE](LICENSE) file in the root directory
of this source tree.

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Check the [FAQ](https://durandtibo.github.io/coola/faq)
- Review existing issues and pull requests

Thank you for contributing to `coola`! 🎉
