import numpy as np

import ndfeb_uvvis as nu
from ndfeb_uvvis.simulate import Design, Effects, pure_spectrum


def test_layout_matches_instrument_export():
    spectra, conc, labels = nu.simulate(20, seed=0)
    assert spectra.shape == (20, 336)
    assert spectra.columns[0] == "230" and spectra.columns[-1] == "900"
    assert list(conc.index) == list(spectra.index) == list(labels.index)
    assert {"Nd (3+)", "Pr (3+)", "Fe (3+)"} <= set(conc.columns)


def test_reproducible_with_seed():
    a = nu.simulate(10, seed=42)[0]
    b = nu.simulate(10, seed=42)[0]
    c = nu.simulate(10, seed=43)[0]
    assert np.array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)


def test_concentrations_within_ranges_and_labels_follow_threshold():
    design = Design()
    _, conc, labels = nu.simulate(200, seed=1, design=design)
    for species, (lo, hi) in design.ranges.items():
        assert conc[species].min() >= 0 and conc[species].max() <= hi
    high = conc["Fe (3+)"] > design.fe_threshold
    assert (labels[high] == "High").all() and (labels[~high] == "Low").all()
    assert 0 < high.mean() < 1  # both classes present


def test_noise_free_spectra_obey_beer_lambert():
    no_effects = Effects(noise_sd=0, offset_sd=0, slope_sd=0, shift_sd=0, saturation=None)
    spectra, conc, _ = nu.simulate(5, seed=2, effects=no_effects)
    wl = spectra.columns.to_numpy(float)
    expected = sum(np.outer(conc[s], pure_spectrum(s, wl)) for s in conc.columns)
    assert np.allclose(spectra.values, expected)


def test_saturation_clips_absorbance():
    spectra, _, _ = nu.simulate(50, seed=3, effects=Effects(saturation=2.0))
    assert spectra.values.max() <= 2.0
