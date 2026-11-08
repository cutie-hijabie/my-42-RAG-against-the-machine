"""Provide the command-line entry point for the RAG project.

Python runs this module when the package is launched with
``uv run python -m src``. Python Fire exposes the ``Cli`` class so
users can select a command and pass its arguments from the terminal.
"""

import fire
from src.cli import Cli

if __name__ == "__main__":
    fire.Fire(Cli)
