"""Setuptools shim — all real metadata lives in ``pyproject.toml``.

This file exists only so that older versions of pip (< 21.3, which do not
yet implement PEP 660) can still run ``pip install -e .`` in editable
mode. Modern pip ignores it and reads everything from ``pyproject.toml``.
"""

from setuptools import setup

setup()
