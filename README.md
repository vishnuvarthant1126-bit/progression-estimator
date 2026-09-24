# Progression Estimator

Predicts a patient's diabetes progression score one year after baseline from ten routine measurements, explains every prediction, and shows the model registry that chose the model. The demo is a single static page that runs the exported model in the browser with no backend.

**Live demo:** https://vishnuvarthant1126-bit.github.io/progression-estimator/

## What's in the repo

```
src/progression/
  data.py       fixed train / holdout split, data fingerprint
  models.py     six candidate models, trained in order
  evaluate.py   repeated k-fold CV, holdout metrics, conformal interval
  gate.py       promotion rules (improvement, parsimony, regression)
  registry.py   file-based registry: every version kept, champion pointer
  explain.py    per-feature contributions that sum to the prediction
  export.py     champion → JSON the page executes
  train.py      CLI that runs all of the above
registry/registry.json   every trained version, metrics and gate decision
web/template.html        the demo page; the model JSON is injected at build
web/model.json           exported champion payload
index.html               built page served by GitHub Pages
tests/                   26 tests, including Python ↔ JavaScript parity
```

## How a model becomes champion

1. **One holdout, set aside once.** 89 of the 442 patients are split off with a fixed seed. They are scored for the record but never used to choose, tune or compare models.
2. **Cross-validation on the rest.** Every candidate is scored by 5-fold cross-validation repeated 3 times on the 353 training patients. Hyperparameter search for gradient boosting runs inside each training fold (nested), so tuning never sees the fold it is scored on.
3. **Promotion gate** (`gate.py`), applied to each candidate against the current champion:
   * promote if CV RMSE improves by at least 0.25;
   * within that tie band, promote if it uses fewer input features (same accuracy, simpler model);
   * flag as a **regression** if CV RMSE is more than 3% worse or CV R² drops by more than 0.02;
   * otherwise reject and keep the champion.

## Results

| Version | Model | Inputs used | CV RMSE (± sd) | CV R² | Holdout RMSE | Gate |
|---|---|---|---|---|---|---|
| v1 | linear regression | 10 | 54.93 ± 3.09 | 0.493 | 53.85 | promoted |
| v2 | ridge | 10 | 55.15 ± 3.00 | 0.490 | 53.78 | rejected |
| v3 | lasso | 7 | 55.15 ± 2.99 | 0.490 | 52.92 | **champion** |
| v4 | random forest | 10 | 58.61 ± 2.76 | 0.422 | 53.10 | regression |
| v5 | gradient boosting (default) | 10 | 60.89 ± 3.60 | 0.376 | 54.06 | regression |
| v6 | gradient boosting (tuned) | 10 | 56.79 ± 2.87 | 0.457 | 51.85 | regression |

Champion on the holdout: RMSE 52.92, MAE 42.79, R² 0.471.

**The main finding.** On the single holdout split, tuned gradient boosting looks best (51.85). Across 15 cross-validation folds it is clearly worse than the linear models (56.79 vs 55.15). The earlier version of this project picked gradient boosting from one split; the pipeline now shows that result was split luck. With 353 patients and ten mostly linear signals, a sparse linear model is as accurate as anything tried and far easier to explain. Lasso drops age, LDL and the cholesterol ratio entirely.

## Prediction interval

The page shows a 90% range around every prediction, built with split conformal prediction from out-of-fold cross-validation residuals: ±95 points. On the untouched holdout it covers 93% of patients. The range is wide on purpose: it is an honest statement of how much one-year progression varies between patients with the same measurements.

## Explanations

For the linear champion, each input's contribution is `coef × (value − training mean)`, so the bars start from the average training patient and add up exactly to the prediction. For gradient-boosting models the exporter uses path attribution (Saabas), which has the same additive property. `explain.py` and the page's JavaScript implement the same arithmetic, and the tests check both.

## Tests

```
pip install -r requirements.lock && pip install -e . --no-deps
pytest -q                               # 26 tests, under a second
python -m progression.train --check     # retrains everything, must reproduce the registry exactly
```

The suite covers:

* **Export parity.** Exported linear and gradient-boosting models reproduce scikit-learn to 1e-6.
* **JavaScript parity.** The inference code inside the page runs in Node and must match Python on every holdout patient.
* **Explanations.** Contributions sum to the prediction.
* **Leakage and determinism.** The holdout never overlaps training, and every registry version was trained on the same data fingerprint.
* **Gate rules.** Promotion, parsimony, rejection and regression cases.
* **Quality floor.** CI fails if a retrain produces a champion with CV RMSE above 57 or R² below 0.45.
* **Interval calibration.** Holdout coverage must stay at or above 85%.

GitHub Actions runs all of it on every push, with pinned library versions so the registry check is exact.

## Retraining

```
python -m progression.train             # rebuilds registry, web/model.json and index.html
python -m progression.train --build-only   # re-renders index.html after editing web/template.html
```

## Data

Efron, Hastie, Johnstone & Tibshirani (2004), *Least Angle Regression*: 442 patients, ten baseline variables, and a quantitative disease-progression measure one year later, loaded through `sklearn.datasets.load_diabetes(scaled=False)` so the inputs are in their original clinical units. The source codes sex as 1 or 2 without documenting which is which, and records serum triglycerides only as a log value, so the page shows both as recorded.

## Disclaimer

A technical demonstration, not a medical device or clinical decision support. The reference bands on the page are general adult screening guidelines. Predictions come from a research dataset and must not be used to make decisions about anyone's health.

## Licence

Code: MIT (see `LICENSE`). Font: Departure Mono by Helena Zhang, SIL Open Font License 1.1.
