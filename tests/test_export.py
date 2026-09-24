"""The exported JSON must reproduce scikit-learn exactly, and explanations must add up."""
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from progression import explain
from progression.data import FEATURES
from progression.export import export_model


def wrap(model_json):
    return {"model": model_json, "features": [{"name": f[0]} for f in FEATURES]}


@pytest.mark.parametrize("model", [
    Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
    Pipeline([("scale", StandardScaler()), ("model", LassoCV(cv=5, random_state=0))]),
    GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=0),
    GradientBoostingRegressor(n_estimators=80, max_depth=2, subsample=0.7, random_state=1),
])
def test_exported_model_matches_sklearn(ds, model):
    model.fit(ds.X_train, ds.y_train)
    ex = wrap(export_model(model, ds.X_train))
    ours = np.array([explain.predict(ex, x) for x in ds.X_test])
    np.testing.assert_allclose(ours, model.predict(ds.X_test), rtol=0, atol=1e-6)


@pytest.mark.parametrize("model", [
    Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
    GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=0),
])
def test_contributions_sum_to_prediction(ds, model):
    model.fit(ds.X_train, ds.y_train)
    ex = wrap(export_model(model, ds.X_train))
    for x, y in zip(ds.X_test[:25], model.predict(ds.X_test[:25])):
        base, c = explain.contributions(ex, x)
        assert base + c.sum() == pytest.approx(y, abs=1e-6)


def test_linear_contribution_is_zero_at_training_mean(ds):
    model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(ds.X_train, ds.y_train)
    ex = wrap(export_model(model, ds.X_train))
    _, c = explain.contributions(ex, ds.X_train.mean(axis=0))
    np.testing.assert_allclose(c, 0, atol=1e-9)


def test_unsupported_champion_is_refused(ds):
    rf = RandomForestRegressor(n_estimators=5, random_state=0).fit(ds.X_train, ds.y_train)
    with pytest.raises(TypeError):
        export_model(rf, ds.X_train)
