import numpy as np

from progression import data


def test_split_is_deterministic(ds):
    again = data.load()
    assert ds.fingerprint == again.fingerprint
    np.testing.assert_array_equal(ds.X_test, again.X_test)


def test_holdout_does_not_overlap_training(ds):
    train_rows = {tuple(r) for r in ds.X_train}
    assert not any(tuple(r) in train_rows for r in ds.X_test)
    assert len(ds.y_train) + len(ds.y_test) == 442


def test_features_are_in_clinical_units(ds):
    # load_diabetes(scaled=False): real ages and BMI, not mean-centred values
    age, bmi = ds.X_train[:, 0], ds.X_train[:, 2]
    assert 18 <= age.min() and age.max() <= 80
    assert 15 <= bmi.min() and bmi.max() <= 45
