from progression import gate


def entry(rmse, r2=0.48, n=10):
    return {"cv_rmse": rmse, "cv_r2": r2, "n_features_used": n}


def test_first_model_becomes_champion():
    d = gate.decide(entry(55), None)
    assert d.promote and d.status == "promoted"


def test_clear_improvement_is_promoted():
    assert gate.decide(entry(54.0), entry(55.0)).promote


def test_near_tie_keeps_champion():
    d = gate.decide(entry(54.9), entry(55.0))
    assert not d.promote and d.status == "rejected"


def test_near_tie_with_fewer_features_is_promoted():
    d = gate.decide(entry(55.1, n=7), entry(55.0, n=10))
    assert d.promote and "simpler" in d.reason


def test_worse_model_is_flagged_as_regression():
    d = gate.decide(entry(58.0), entry(55.0))
    assert not d.promote and d.status == "regression"


def test_r2_drop_is_flagged_even_if_rmse_close():
    d = gate.decide(entry(55.5, r2=0.44), entry(55.0, r2=0.48))
    assert d.status == "regression"
