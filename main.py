#!/usr/bin/env python3
"""Run the assistant from a source checkout: ``python main.py`` (same as ``agentic-assistant``)."""

import sys

from agentic_assistant.cli import main

if __name__ == "__main__":
    sys.exit(main())
