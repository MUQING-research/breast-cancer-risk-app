# Data preprocessing

This folder contains training-only EDA, transformation screening, functional-form checks, and design-matrix construction. All distribution-dependent decisions are fitted on the training split and exported for model rebuilding.

The preprocessing implementation is `eda_and_preprocessing.py`. It is called by the model training stage and is not part of the Shiny deployment package.
