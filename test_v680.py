"""Expose the standalone V680 tests to the repository-wide unittest command."""
from __future__ import annotations

import pathlib
import sys
import unittest
import importlib


V680_DIRECTORY = pathlib.Path(__file__).parent / "research" / "v680"
sys.path.insert(0, str(V680_DIRECTORY))


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for path in sorted(V680_DIRECTORY.glob("test_*.py")):
        suite.addTests(loader.loadTestsFromModule(importlib.import_module(path.stem)))
    return suite
