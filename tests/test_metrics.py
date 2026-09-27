import numpy as np

from ndfeb_uvvis import regression_report


def test_perfect_prediction():
    r = regression_report([1, 2, 3], [1, 2, 3])
    assert r["RMSEP"] == 0 and r["Bias"] == 0 and r["R2"] == 1


def test_constant_bias_separated_from_sep():
    y = np.array([0.1, 0.2, 0.3, 0.4])
    r = regression_report(y, y + 0.01)
    assert np.isclose(r["Bias"], 0.01)
    assert np.isclose(r["RMSEP"], 0.01)
    assert np.isclose(r["SEP"], 0.0)


def test_relative_error_ignores_zero_references():
    r = regression_report([0.0, 0.1], [0.05, 0.11])
    assert np.isclose(r["Mean rel. error (%)"], 10.0)
