# Model training and evaluation

This folder contains the offline training pipeline, EDA helpers, model-bundle export command, and model/visualization tests.

Install the training dependencies from the repository root:

```bash
python -m pip install -r 03_model_training_and_evaluation/requirements.txt
python 03_model_training_and_evaluation/train_and_export_model_bundle.py
python -m unittest discover -s 03_model_training_and_evaluation -p "test_*.py" -v
```

Five-fold cross-validation is used for LASSO hyperparameter selection as internal validation. The fixed 20% holdout is reserved for test-set evaluation and is not used for tuning. The deployment threshold is selected by the Youden index on training predictions and locked before test evaluation. `training_metrics.json` is the canonical metric manifest for the current bundle. Run `python repeated_validation.py` to perform a separate 5-fold, 10-repeat conditional audit with fold-level variability, fold-local Youden thresholds, patient-level bootstrap confidence intervals, calibration metrics, and fixed-penalty LASSO selection frequencies. Training outputs are written to `03_model_training_and_evaluation/.cache/training/`. The generated deployment bundle and preprocessing decision file are written to `04_model_deployment/` so the deployment package can be tested locally before upload.
