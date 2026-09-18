# Data cleaning

This folder contains the base cleaning stage for the breast-cancer dataset. It standardizes the source table, records the data dictionary and audit, and writes the cleaned table for downstream preprocessing.

Run from the repository root:

```bash
python 01_data_cleaning/clean_dataset.py
```

Outputs are written to `03_model_training_and_evaluation/.cache/training/` and are excluded from deployment and Git.
