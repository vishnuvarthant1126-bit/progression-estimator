# Progression Estimator

A gradient-boosting regression model that predicts one-year disease progression from routine clinical measurements — served as a single self-contained HTML page that runs inference in the browser, with no backend.

**Live demo:** https://vishnuvarthant1126-bit.github.io/progression-estimator/

## What it does

You adjust a patient's measurements — demographics, vitals, lipid panel and metabolic markers — and the page returns a predicted one-year progression score in real time, along with where that score falls within the population range of the target variable.

Inputs are grouped and annotated with general adult screening reference ranges and colour-coded by zone (healthy, borderline, higher risk), so the panel reads like a set of lab results rather than a row of anonymous sliders. Values are converted under the hood into the standardised feature space the model was trained on, so the prediction always reflects what is displayed.

## How inference works

The trained ensemble is exported to JSON — every tree's split features, thresholds, child pointers and leaf values — and embedded directly in the page. At runtime the page walks each tree for the current input vector and accumulates the result:

```
prediction = init_value + learning_rate * Σ tree_k(x)
```

There is no API call and no server process. The model that produced the reported metrics is the exact model executing in the browser, which means the page runs offline and deploys as a static file.

## Champion model

| | |
| --- | --- |
| Model | Gradient-boosting regressor |
| Version | v4 (champion) |
| Learning rate | 0.05 |
| RMSE | 53.05 |
| MAE | 43.36 |
| R² | 0.469 |
| MAPE | 39.2% |

## Model registry and regression testing

Each trained candidate is recorded in a version registry rather than overwriting the one before it. The page surfaces that history directly: every version, the score it achieved, and which one currently holds champion status.

Promotion is gated. A regression-detection test suite compares each challenger against the reigning champion and blocks promotion when the challenger degrades on the tracked metrics — one candidate in the version history is flagged for exactly that reason. The point is that "the newest model wins" is never the default; a model has to earn the champion slot.

## Running it locally

A single static file, no build step and no dependencies:

```
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Deploying

### GitHub Pages

1. Push this repository to GitHub.
2. Go to Settings → Pages, set Source to `main` and folder to `/ (root)`.
3. The page is served at `https://<username>.github.io/<repository>/`.

### Vercel

```
npx vercel --prod
```

Or import the repository at vercel.com — no framework preset and no build command are required.

## Notes

The fitted estimator is exported tree-by-tree into the JSON payload inside `index.html`. Retraining means re-running the training pipeline and replacing that payload; nothing else in the page needs to change.

## Disclaimer

This is a technical demonstration, not a medical device and not clinical decision support. The reference ranges shown are general adult screening guidelines and vary with age, sex, medical history, fasting status and overall cardiovascular risk. Predictions are model output on a research dataset and must not be used to make decisions about any person's health.
