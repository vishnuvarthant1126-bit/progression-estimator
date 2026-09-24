"""Train every candidate, gate promotions, update the registry, export the champion.

    python -m progression.train            # rebuild registry + web/model.json + index.html
    python -m progression.train --build-only   # re-render index.html from web/model.json
    python -m progression.train --check    # verify the committed registry is reproducible
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import data, evaluate, gate
from .export import build_payload
from .models import CANDIDATES
from .registry import DEFAULT_PATH, Registry

ROOT = Path(__file__).resolve().parents[2]
WEB_MODEL = ROOT / "web" / "model.json"
TEMPLATE = ROOT / "web" / "template.html"
INDEX = ROOT / "index.html"
PLACEHOLDER = "/*__MODEL_JSON__*/"


def n_features_used(est) -> int:
    """Inputs the fitted model actually depends on (non-zero coefficients for linear models)."""
    est = getattr(est, "best_estimator_", est)
    model = est.named_steps["model"] if hasattr(est, "named_steps") else est
    if hasattr(model, "coef_"):
        return int(np.count_nonzero(np.abs(model.coef_) > 1e-9))
    if hasattr(model, "feature_importances_"):
        return int(np.count_nonzero(model.feature_importances_ > 0))
    return len(data.FEATURE_NAMES)


def run(verbose: bool = True) -> tuple[Registry, dict, object]:
    ds = data.load()
    reg = Registry()
    fitted = {}
    for cand in CANDIDATES:
        est = cand.build()
        cv = evaluate.cross_validate(est, ds.X_train, ds.y_train)
        est.fit(ds.X_train, ds.y_train)
        pred = est.predict(ds.X_test)
        hold = evaluate.metrics(ds.y_test, pred)
        entry = {
            "name": cand.name, "family": cand.family, "description": cand.description,
            **{k: round(v, 4) for k, v in cv.items()},
            **{f"holdout_{k}": round(v, 4) for k, v in hold.items()},
            "holdout_interval_coverage": round(evaluate.interval_coverage(ds.y_test, pred, cv["interval_halfwidth"]), 4),
            "n_features_used": n_features_used(est),
            "data_fingerprint": ds.fingerprint,
        }
        decision = gate.decide(entry, reg.champion_entry)
        entry["status"], entry["reason"] = decision.status, decision.reason
        entry = reg.register(entry, promote=decision.promote)
        fitted[entry["version"]] = est
        if verbose:
            print(f"{entry['version']:>3}  {cand.name:<27} CV RMSE {cv['cv_rmse']:6.2f} ± {cv['cv_rmse_std']:4.2f}"
                  f"  holdout {hold['rmse']:6.2f}  -> {decision.status}")
    for v in reg.versions:
        if v["version"] == reg.champion:
            v["status"] = "champion"
    champ = reg.champion_entry
    payload = build_payload(fitted[reg.champion], champ, reg.versions, ds)
    return reg, payload, fitted[reg.champion]


def build_index(payload: dict) -> None:
    html = TEMPLATE.read_text()
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    INDEX.write_text(html.replace(PLACEHOLDER, blob))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-only", action="store_true", help="rebuild index.html from web/model.json without training")
    ap.add_argument("--check", action="store_true", help="fail if retraining changes the committed registry")
    args = ap.parse_args(argv)
    if args.build_only:
        build_index(json.loads(WEB_MODEL.read_text()))
        print("index.html rebuilt")
        return 0
    reg, payload, _ = run()
    if args.check:
        committed = Registry.load(DEFAULT_PATH)
        if committed.versions != reg.versions or committed.champion != reg.champion:
            print("Registry is not reproducible from the current code.", file=sys.stderr)
            return 1
        print("Registry reproduced exactly.")
        return 0
    reg.save(DEFAULT_PATH)
    WEB_MODEL.write_text(json.dumps(payload, indent=1) + "\n")
    if TEMPLATE.exists():
        build_index(payload)
    print(f"Champion: {reg.champion} ({reg.champion_entry['name']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
