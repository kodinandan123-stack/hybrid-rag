# Empty on purpose. Its presence makes pytest treat the repository root as an
# import root (pytest inserts the directory containing the outermost
# conftest.py onto sys.path), so first-party top-level packages such as
# retrieval/, ingestion/, generation/, config/, and api/ are importable by
# the test suite regardless of how pytest is invoked (bare `pytest`, not
# just `python -m pytest`). Without this file, CI's `pytest -v` step fails
# every test module with ModuleNotFoundError, since running the console
# script (unlike `python -m pytest`) never adds the current directory to
# sys.path on its own.
