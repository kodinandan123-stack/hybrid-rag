# Contributing to Hybrid RAG

Thank you for considering a contribution. This document outlines the workflow and standards for submitting changes.

## Development Setup

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

3. Copy `.env.example` to `.env` and fill in the required values.

## Branching Convention

| Branch prefix | Purpose |
|---------------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring, no behaviour change |
| `test/` | Test additions or fixes |

Always branch off `main` and keep branches short-lived.

## Running Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a pull request.

## Code Style

- Follow PEP 8. Use `ruff` for linting and `black` for formatting.
- Run `ruff check . && black --check .` before committing.
- Type-annotate all public functions and methods.
- Write docstrings for every public class and function.

## Pull Request Process

1. Open a PR against `main` with a clear title and description.
2. Reference any related issues with `Closes #<issue>`.
3. Ensure CI passes (tests, lint, type checks).
4. Request a review from a maintainer.
5. Squash commits before merge if the history is noisy.

## Commit Message Format

Use the imperative mood and keep the subject line under 72 characters:

```
Step N: <short description>

[Optional body explaining the why]
```

## Reporting Issues

Open a GitHub Issue with:
- A clear, descriptive title.
- Steps to reproduce the problem.
- Expected vs. actual behaviour.
- Environment details (OS, Python version, dependency versions).
