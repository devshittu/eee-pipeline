# src/main.py
# File path: src/main.py

"""
DEPRECATED: This entry point is maintained for backward compatibility.
Please use the new CLI interface instead:

    python -m src.cli.main <command>

Or install as a package and use:

    eee-cli <command>

Examples:
    python -m src.cli.main documents process "Some text"
    python -m src.cli.main documents batch input.jsonl --output results.jsonl
    python -m src.cli.main jobs status <job-id>
    python -m src.cli.main admin health

For migration assistance, see documentation.
"""

from src.cli.main import cli
import sys
import warnings

warnings.warn(
    "src/main.py is deprecated. Use 'python -m src.cli.main' or 'eee-cli' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import and delegate to the new CLI

if __name__ == "__main__":
    cli(obj={})


# src/main.py
# File path: src/main.py
