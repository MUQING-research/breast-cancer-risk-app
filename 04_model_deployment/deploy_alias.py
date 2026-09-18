# Module guide:
# - Role: Validate and publish the Shiny application.
# - Workflow: Check required runtime files before delegating to the deployment command.
# - Design note: Fail early when package contents or runtime compatibility are invalid.
"""Backward-compatible entry point for the shinyapps.io deployment helper."""

import sys
from pathlib import Path

SHINYAPP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHINYAPP_DIR))

from deploy import main


if __name__ == "__main__":
    raise SystemExit(main())
