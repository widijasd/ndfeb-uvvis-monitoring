import numpy as np
import pytest

import ndfeb_uvvis as nu

COMPONENTS = {"Nd (3+)": 1, "Pr (3+)": 3}
WINDOWS = nu.PUBLISHED_WAVELENGTHS


@pytest.fixture(scope="module")
def data():
    cal = nu.simulate(200, seed=0)
    test = nu.simulate(50, seed=1)
    models = nu.fit_models(*cal, n_components=COMPONENTS, selected_wavelengths=WINDOWS)
    return cal, test, models


def test_preprocessing_range_cut():
    wl = np.arange(230.0, 902.0, 2.0)
    pipe = nu.build_preprocessing(wl).fit(np.random.default_rng(0).normal(size=(5, wl.size)))
    cut = pipe.named_steps["rangecut"].x_axis_
    # chemotools RangeCut keeps the start wavelength; whether the end point is kept
    # depends on the grid (it is excluded here, giving 400-898 nm), so only bound it.
    assert cut[0] == 400
    assert 896 <= cut[-1] <= 902
    assert np.all(np.diff(cut) > 0) and np.isin(cut, wl).all()


def test_predicts_independent_test_set(data):
    _, (spectra, conc, labels), models = data
    pred = nu.predict(spectra, models)
    table = nu.validation_table(conc, pred, COMPONENTS)
    for target in COMPONENTS:
        span = conc[target].max() - conc[target].min()
        assert table.loc[target, "RMSEP"] < 0.05 * span, table
        assert table.loc[target, "R2"] > 0.98
    assert (pred["Fe class"].to_numpy() == labels.to_numpy()).mean() >= 0.95


def test_save_load_roundtrip(tmp_path, data):
    _, (spectra, _, _), models = data
    path = nu.save_models(models, tmp_path / "models.joblib")
    assert np.allclose(nu.predict(spectra, path).iloc[:, :3], nu.predict(spectra, models).iloc[:, :3])


def test_rejects_wrong_wavelengths(data):
    _, (spectra, _, _), models = data
    with pytest.raises(ValueError, match="calibration wavelengths"):
        nu.predict(spectra.iloc[:, :-1], models)


def test_selected_wavelengths_mask(data):
    (spectra, conc, labels), _, _ = data
    sel = {"Nd (3+)": np.arange(560.0, 600.0, 2.0)}
    models = nu.fit_models(spectra, conc, n_components={"Nd (3+)": 2}, selected_wavelengths=sel)
    assert models["regressors"]["Nd (3+)"]["mask"].sum() == 20
    assert "fe_model" not in models  # no labels given


def test_venetian_blinds_partition():
    cv = nu.VenetianBlinds(4)
    X = np.zeros((10, 1))
    tests = [t for _, t in cv.split(X)]
    assert sorted(np.concatenate(tests)) == list(range(10))
    assert list(tests[0]) == [0, 4, 8]


def test_cross_validation_error_falls_with_components(data):
    (spectra, conc, _), _, models = data
    X = models["preprocessing"].transform(spectra.to_numpy())
    cv = nu.cross_validate_components(X, conc["Nd (3+)"], max_components=4)
    assert cv["RMSECV"].iloc[-1] < cv["RMSECV"].iloc[0]


def test_published_wavelength_selection():
    assert len(nu.PUBLISHED_WAVELENGTHS["Nd (3+)"]) == 125
    assert len(nu.PUBLISHED_WAVELENGTHS["Pr (3+)"]) == 23
    for wl in nu.PUBLISHED_WAVELENGTHS.values():
        assert np.all(np.diff(wl) > 0) and wl.min() >= 400 and wl.max() <= 898
