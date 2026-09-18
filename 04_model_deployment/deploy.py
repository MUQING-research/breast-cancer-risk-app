# Module guide:
# - Role: Validate and publish the Shiny application.
# - Workflow: Check required runtime files before delegating to the deployment command.
# - Design note: Fail early when package contents or runtime compatibility are invalid.
"""Validate the runtime package and deploy the existing shinyapps.io app."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ACCOUNT = "medictio"
TITLE = "breast-cancer-classifier"
APP_ID = "17579058"
ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = (
    ".python-version",
    "app.py",
    "visualizations.py",
    "breast_cancer_app.py",
    "breast_cancer_model_bundle.pkl",
    "preprocessing_decisions.json",
    "compact_theme.css",
    "world.geojson",
    "requirements.txt",
)
LOCAL_UPLOAD_PYTHON_VERSIONS = {(3, 13), (3, 12)}


# Function guide: validate_package is responsible for validate package.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def validate_package() -> None:
    """Require a complete package and the tested Python/dependency versions."""
    missing = [name for name in RUNTIME_FILES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError("Missing runtime files: " + ", ".join(missing))
    if sys.version_info[:2] not in LOCAL_UPLOAD_PYTHON_VERSIONS:
        raise RuntimeError("Deploy with the tested Python 3.12 or 3.13 environment.")
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name, expected = requirement.split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {actual}")


# Function guide: main is responsible for run the module main workflow.
# Inputs: the component state. Outputs and side effects follow the routine contract.
# Keep this boundary focused on one workflow step so it remains easy to test and reuse.
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without uploading.")
    args = parser.parse_args()
    validate_package()
    print("Runtime files: " + ", ".join(RUNTIME_FILES), flush=True)
    if args.check:
        print("Deployment preflight passed.")
        return 0
    # Stage only the runtime package so project caches and deployment metadata
    # directories cannot be interpreted as extra files by rsconnect.
    with tempfile.TemporaryDirectory(prefix="breast_cancer_shiny_deploy_") as temp_dir:
        staging_root = Path(temp_dir)
        for filename in RUNTIME_FILES:
            shutil.copy2(ROOT / filename, staging_root / filename)

        cmd = [
            sys.executable, "-m", "rsconnect.main", "deploy", "shiny",
            str(staging_root),
            "--name", ACCOUNT, "--title", TITLE, "--app-id", APP_ID,
            "--python", sys.executable,
        ]
        # shinyapps.io does not support rsconnect --environment management.
        print(f"Deploying https://{ACCOUNT}.shinyapps.io/{TITLE}/", flush=True)
        return subprocess.run(cmd, cwd=staging_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
