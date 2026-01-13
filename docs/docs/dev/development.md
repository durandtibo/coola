# Development Guide

This guide covers setting up your development environment and common development tasks.

## Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- Git for version control
- Basic knowledge of Python and testing

## Initial Setup

### 1. Fork and Clone

```shell
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/coola.git
cd coola
```

### 2. Install Poetry

If you don't have Poetry installed:

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

Verify installation:

```shell
poetry --version
```

### 3. Install Dependencies

```shell
# Install all dependencies (including dev dependencies)
poetry install --no-interaction

# Or use make command
make install
```

To install with docs dependencies:

```shell
poetry install --with docs
# Or
make install-all
```

### 4. Set Up Pre-commit Hooks

```shell
poetry run pre-commit install
```

This will automatically run code quality checks before each commit.

## Development Workflow

### Running Tests

**Run all unit tests:**

```shell
make unit-test
```

**Run tests with coverage:**

```shell
make unit-test-cov
```

**Run specific test file:**

```shell
poetry run pytest tests/unit/test_comparison.py
```

**Run specific test:**

```shell
poetry run pytest tests/unit/test_comparison.py::test_objects_are_equal
```

**Run tests with verbose output:**

```shell
poetry run pytest -v tests/unit/
```

### Code Quality

**Format code with Black:**

```shell
poetry run black .
# Or
make format
```

**Run linter (Ruff):**

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

### Documentation

**Build documentation locally:**

```shell
poetry run mkdocs serve -f docs/mkdocs.yml
```

Then open http://127.0.0.1:8000 in your browser.

**Build documentation without serving:**

```shell
poetry run mkdocs build -f docs/mkdocs.yml
```

**Run doctests:**

```shell
make doctest-src
```

### Type Checking

While `coola` doesn't currently use mypy in CI, you can run type checking locally:

```shell
poetry run mypy src/coola --ignore-missing-imports
```

## Project Structure

```
coola/
├── .github/               # GitHub configuration
│   ├── workflows/        # CI/CD workflows
│   ├── CONTRIBUTING.md   # Contribution guidelines
│   └── ISSUE_TEMPLATE/   # Issue templates
├── docs/                 # Documentation
│   ├── docs/            # Documentation source
│   └── mkdocs.yml       # MkDocs configuration
├── src/                 # Source code
│   └── coola/
│       ├── comparison.py           # Main API
│       ├── equality/              # Equality comparison
│       │   ├── comparators/      # Type-specific comparators
│       │   ├── testers/          # Comparison testers
│       │   └── handlers/         # Comparison handlers
│       └── utils/                # Utility functions
├── tests/               # Test files
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── pyproject.toml      # Project configuration
├── poetry.lock         # Locked dependencies
├── LICENSE             # License file
├── README.md           # Project README
└── SECURITY.md         # Security policy
```

## Common Development Tasks

### Adding a New Feature

1. **Create a new branch:**
   ```shell
   git checkout -b feature/my-feature
   ```

2. **Implement the feature:**
    - Write code in `src/coola/`
    - Add tests in `tests/unit/`
    - Update documentation in `docs/docs/`

3. **Run tests:**
   ```shell
   make unit-test-cov
   ```

4. **Run code quality checks:**
   ```shell
   poetry run pre-commit run --all-files
   ```

5. **Commit changes:**
   ```shell
   git add .
   git commit -m "Add: brief description of feature"
   ```

6. **Push and create PR:**
   ```shell
   git push origin feature/my-feature
   ```

### Fixing a Bug

1. **Create a branch:**
   ```shell
   git checkout -b fix/bug-description
   ```

2. **Write a failing test** that reproduces the bug

3. **Fix the bug**

4. **Verify the test passes:**
   ```shell
   poetry run pytest tests/unit/path/to/test.py
   ```

5. **Run full test suite:**
   ```shell
   make unit-test-cov
   ```

6. **Commit and push:**
   ```shell
   git commit -m "Fix: description of bug fix"
   git push origin fix/bug-description
   ```

### Adding Support for a New Type

1. **Create a new comparator:**
   ```python
   # src/coola/equality/comparators/mytype_.py
   from typing import Any
   from coola.equality.config import EqualityConfig
   from coola.equality.comparators.base import BaseEqualityComparator


   class MyTypeComparator(BaseEqualityComparator):
       def clone(self) -> "MyTypeComparator":
           return self.__class__()

       def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
           # Type check
           if type(actual) is not type(expected):
               if config.show_difference:
                   # Log difference
                   pass
               return False

           # Implement comparison logic
           return actual == expected
   ```

2. **Register the comparator:**
   ```python
   # In the appropriate __init__.py
   from coola.equality.testers import EqualityTester
   from coola.equality.comparators.mytype_ import MyTypeComparator

   EqualityTester.registry[MyType] = MyTypeComparator()
   ```

3. **Write tests:**
   ```python
   # tests/unit/equality/comparators/test_mytype_.py
   import pytest
   from coola import objects_are_equal


   def test_mytype_equal():
       obj1 = MyType(...)
       obj2 = MyType(...)
       assert objects_are_equal(obj1, obj2)
   ```

4. **Update documentation:**
    - Add to `docs/docs/types.md`
    - Add examples to `docs/docs/examples.md`

### Updating Dependencies

```shell
# Update all dependencies
poetry update

# Update specific dependency
poetry update numpy

# Update poetry itself
poetry self update
```

## Testing Guidelines

### Writing Good Tests

1. **Use descriptive names:**
   ```python
   def test_objects_are_equal_with_identical_dicts_returns_true(): ...


   def test_objects_are_equal_with_different_types_returns_false(): ...
   ```

2. **Test edge cases:**
    - Empty collections
    - None values
    - Large data
    - Deeply nested structures

3. **Use fixtures for common data:**
   ```python
   @pytest.fixture
   def sample_tensor():
       return torch.randn(10, 10)


   def test_tensor_comparison(sample_tensor):
       result = objects_are_equal(sample_tensor, sample_tensor)
       assert result is True
   ```

4. **Test both success and failure cases:**
   ```python
   def test_success_case():
       assert objects_are_equal(obj1, obj2)


   def test_failure_case():
       assert not objects_are_equal(obj1, obj3)
   ```

### Running Tests with Different Configurations

**Run tests for a specific module:**

```shell
poetry run pytest tests/unit/equality/
```

**Run tests matching a pattern:**

```shell
poetry run pytest -k "tensor"
```

**Run tests with markers:**

```shell
poetry run pytest -m "slow"
```

**Run tests in parallel:**

```shell
poetry run pytest -n auto
```

## Continuous Integration

The project uses GitHub Actions for CI. Workflows are in `.github/workflows/`:

- **CI**: Runs on every push and PR
    - Linting
    - Tests
    - Coverage

- **Documentation**: Builds and deploys docs
    - Builds on every push
    - Deploys on release

- **Nightly Tests**: Tests against latest dependencies
    - Runs daily
    - Tests multiple Python versions

## Release Process

Releases are managed by the maintainers:

1. Update version in `pyproject.toml`
2. Create and push a git tag
3. GitHub Actions automatically publishes to PyPI
4. Documentation is automatically deployed

## Getting Help

- **Documentation**: https://durandtibo.github.io/coola/
- **GitHub Issues**: https://github.com/durandtibo/coola/issues
- **Contributing Guide
  **: [CONTRIBUTING.md](https://github.com/durandtibo/coola/blob/main/.github/CONTRIBUTING.md)

## Best Practices

1. **Write tests first** (TDD approach when possible)
2. **Keep PRs focused** on a single feature or fix
3. **Update documentation** for user-facing changes
4. **Run pre-commit hooks** before committing
5. **Write clear commit messages**
6. **Add docstrings** to all public APIs
7. **Keep dependencies minimal**
8. **Follow existing code style**

## Troubleshooting Development Issues

### Test Issues

**Tests fail after pulling changes:**

```shell
# Update dependencies
poetry install
# Re-run tests
make unit-test
```

**Import errors in tests:**

```shell
# Make sure package is installed in development mode
poetry install
```

### Pre-commit Issues

**Pre-commit hooks fail:**

```shell
# Update pre-commit hooks
poetry run pre-commit autoupdate
# Try running manually
poetry run pre-commit run --all-files
```

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] All tests pass locally
- [ ] Code coverage is maintained or improved
- [ ] Pre-commit hooks pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Code follows project style
- [ ] No unnecessary dependencies added
- [ ] Examples are provided for new features
- [ ] Edge cases are tested

## Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
