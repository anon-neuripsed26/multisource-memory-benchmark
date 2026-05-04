"""Allow ``python -m survey2agent`` invocation."""

from __future__ import annotations

import sys

from survey2agent.cli import main


if __name__ == "__main__":
    sys.exit(main())
