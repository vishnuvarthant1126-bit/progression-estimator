"""Checks on the committed registry, payload and page (no retraining needed)."""
import json
import re
from pathlib import Path

from progression import explain

ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_exactly_one_champion(registry_json):
    champs = [v for v in registry_json["versions"] if v["status"] == "champion"]
    assert len(champs) == 1 and champs[0]["version"] == registry_json["champion"]


def test_champion_was_never_beaten_on_cv(registry_json):
    """No non-champion is clearly better on cross-validation (the gate's own rule)."""
    champ = next(v for v in registry_json["versions"] if v["status"] == "champion")
    for v in registry_json["versions"]:
        assert v["cv_rmse"] > champ["cv_rmse"] - 0.25


def test_every_version_was_trained_on_the_same_split(registry_json, ds):
    assert {v["data_fingerprint"] for v in registry_json["versions"]} == {ds.fingerprint}


def test_champion_quality_floor(payload):
    """Regression guard: fail CI if a retrain produces a clearly worse champion."""
    c = payload["champion"]
    assert c["cv_rmse"] < 57.0
    assert c["cv_r2"] > 0.45
    assert c["holdout_r2"] > 0.40


def test_prediction_interval_is_calibrated(payload):
    assert payload["champion"]["holdout_interval_coverage"] >= 0.85


def test_payload_examples_match_exported_model(payload):
    for ex in payload["examples"]:
        assert abs(explain.predict(payload, ex["features"]) - ex["predicted"]) < 1e-6


def test_index_embeds_current_payload(payload):
    html = (ROOT / "index.html").read_text()
    m = re.search(r'<script id="model" type="application/json">(.*?)</script>', html, re.S)
    assert m, "model payload missing from index.html; run python -m progression.train --build-only"
    assert json.loads(m.group(1).replace("<\\/", "</")) == payload
