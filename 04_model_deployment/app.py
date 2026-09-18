# Module guide:
# - Role: Provide the lightweight application entry point used by the Shiny runtime.
# - Workflow: Import the fully configured application object from the neighboring module.
# - Design note: Keep this wrapper side-effect free so deployment tools can discover `app` reliably.
from breast_cancer_app import app as app  # noqa: PLC0414
