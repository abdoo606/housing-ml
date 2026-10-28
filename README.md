# 24 · housing

A regression pipeline in NumPy: encoding, ridge, k-fold CV.

## Run

```bash
from housing.pipeline import train, cross_validate
```

## Test

```bash
pytest projects/24-housing-ml -q
```

## What it demonstrates

- Ridge solved in closed form; intercept left unpenalised by centring features
- One-hot and standardisation fitted on the training fold only - no leakage
- k-fold cross-validation with per-fold metrics
- MAE / RMSE / R² reported together
