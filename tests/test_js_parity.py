"""Run the page's own JavaScript inference in Node and compare with Python/scikit-learn."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from progression import explain
from progression.data import FEATURES
from progression.export import export_model

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def js_predict(model_payload: dict, rows) -> list[float]:
    html = (ROOT / "web" / "template.html").read_text()
    code = re.search(r"// <inference>(.*?)// </inference>", html, re.S).group(1)
    script = code + f"""
var M = {json.dumps(model_payload)};
var rows = {json.dumps([list(map(float, r)) for r in rows])};
console.log(JSON.stringify(rows.map(function(r){{ return predict(M, r); }})));
"""
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_js_matches_python_for_champion(payload, ds):
    js = js_predict(payload, ds.X_test)
    py = [explain.predict(payload, x) for x in ds.X_test]
    np.testing.assert_allclose(js, py, atol=1e-6)


def test_js_matches_sklearn_for_gradient_boosting(ds):
    gbm = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=0).fit(ds.X_train, ds.y_train)
    m = {"model": export_model(gbm, ds.X_train), "features": [{"name": f[0]} for f in FEATURES]}
    np.testing.assert_allclose(js_predict(m, ds.X_test), gbm.predict(ds.X_test), atol=1e-6)
