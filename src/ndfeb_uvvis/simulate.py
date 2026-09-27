"""Synthetic UV-Vis spectra of rare-earth chloride leachates.

The real calibration data behind the published models cannot be shared, so this
module generates realistic stand-in data with the same layout:

* spectra: DataFrame, one row per sample, one column per wavelength (230-900 nm, 2 nm steps)
* concentrations: DataFrame with columns such as ``"Nd (3+)"`` (mol/L)
* labels: Series of ``"High"`` / ``"Low"`` Fe(III) class

Spectra follow Beer-Lambert additivity (1 cm path length): each species contributes
``epsilon(lambda) * c``, with epsilon built from Gaussian bands. Band positions and
molar absorptivities are approximate, literature-typical values for the aquo/chloro
ions. They are illustrative only and were NOT fitted to the unpublished calibration data.

Measurement effects that make the problem realistic are added on top: random noise,
a per-sample baseline offset and slope, small wavelength shifts and detector
saturation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Default wavelength grid (nm), matching the instrument export format.
WAVELENGTHS = np.arange(230.0, 902.0, 2.0)

# (centre nm, molar absorptivity L/mol/cm, half-width nm)
BANDS: dict[str, list[tuple[float, float, float]]] = {
    "Nd (3+)": [
        (354.0, 3.0, 4.0),
        (521.0, 4.5, 5.0),
        (575.0, 7.0, 6.0),
        (740.0, 6.5, 6.0),
        (794.0, 11.0, 7.0),
        (865.0, 3.0, 8.0),
    ],
    "Pr (3+)": [
        (444.0, 10.0, 4.0),
        (468.0, 4.5, 4.0),
        (482.0, 4.5, 4.0),
        (588.0, 1.8, 5.0),
    ],
    # Fe(III) chloro-complexes: intense charge-transfer bands in the UV with a tail into the visible
    "Fe (3+)": [
        (335.0, 2000.0, 25.0),
        (365.0, 1000.0, 25.0),
    ],
    # Co(II) aquo ion: broad d-d band, a realistic interferent overlapping Nd 521/575 nm
    "Co (2+)": [
        (511.0, 5.0, 30.0),
    ],
}

#: Default concentration ranges (mol/L) sampled for each species.
RANGES: dict[str, tuple[float, float]] = {
    "Nd (3+)": (0.0, 0.15),
    "Pr (3+)": (0.0, 0.20),
    "Fe (3+)": (0.0, 0.15),
    "Co (2+)": (0.0, 0.05),
}


def pure_spectrum(species: str, wavelengths: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """Molar absorptivity (L/mol/cm) of one species on the given wavelength grid."""
    wl = np.asarray(wavelengths, dtype=float)
    eps = np.zeros_like(wl)
    for centre, height, width in BANDS[species]:
        eps += height * np.exp(-0.5 * ((wl - centre) / width) ** 2)
    return eps


@dataclass
class Effects:
    """Size of the measurement artefacts added to each simulated spectrum."""

    noise_sd: float = 0.002          # absorbance units, white noise
    offset_sd: float = 0.01          # baseline offset per sample
    slope_sd: float = 1e-5           # baseline slope per sample (AU/nm)
    shift_sd: float = 0.3            # wavelength shift per sample (nm)
    saturation: float | None = 10.0  # detector ceiling; None disables clipping


@dataclass
class Design:
    """How sample concentrations are drawn."""

    ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(RANGES))
    p_absent: float = 0.4            # probability each species is absent from a sample
    p_high_fe: float = 0.2           # fraction of samples deliberately given high Fe(III)
    fe_threshold: float = 0.005      # mol/L; above this a sample is labelled "High"


def simulate(
    n_samples: int = 250,
    *,
    seed: int | None = 0,
    wavelengths: np.ndarray = WAVELENGTHS,
    design: Design | None = None,
    effects: Effects | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Simulate a calibration or test set.

    Returns ``(spectra, concentrations, labels)`` with matching row index.
    """
    design = design or Design()
    effects = effects or Effects()
    rng = np.random.default_rng(seed)
    wl = np.asarray(wavelengths, dtype=float)
    species = list(design.ranges)

    # Concentrations
    conc = np.zeros((n_samples, len(species)))
    for j, sp in enumerate(species):
        lo, hi = design.ranges[sp]
        conc[:, j] = rng.uniform(lo, hi, n_samples)
        conc[rng.random(n_samples) < design.p_absent, j] = 0.0
    if "Fe (3+)" in species:
        j = species.index("Fe (3+)")
        high = rng.random(n_samples) < design.p_high_fe
        lo, hi = design.ranges["Fe (3+)"]
        conc[:, j] = np.where(
            high,
            rng.uniform(max(design.fe_threshold * 2, lo), hi, n_samples),
            rng.uniform(0.0, design.fe_threshold / 2, n_samples) * (rng.random(n_samples) > 0.5),
        )

    # Spectra: Beer-Lambert sum, each sample on a slightly shifted axis
    X = np.empty((n_samples, wl.size))
    for i in range(n_samples):
        wl_i = wl + rng.normal(0.0, effects.shift_sd)
        x = np.zeros_like(wl)
        for j, sp in enumerate(species):
            if conc[i, j] > 0:
                x += pure_spectrum(sp, wl_i) * conc[i, j]
        x += rng.normal(0.0, effects.offset_sd) + rng.normal(0.0, effects.slope_sd) * (wl - wl[0])
        x += rng.normal(0.0, effects.noise_sd, wl.size)
        X[i] = x
    if effects.saturation is not None:
        X = np.minimum(X, effects.saturation)

    index = pd.Index([f"Sample{i + 1}" for i in range(n_samples)], name="Sample")
    columns = [f"{w:g}" for w in wl]  # same as the instrument CSV headers
    spectra = pd.DataFrame(X, index=index, columns=columns)
    concentrations = pd.DataFrame(conc, index=index, columns=species)
    if "Fe (3+)" in species:
        fe = concentrations["Fe (3+)"]
        labels = pd.Series(np.where(fe > design.fe_threshold, "High", "Low"), index=index, name="labels")
    else:
        labels = pd.Series("Low", index=index, name="labels")
    return spectra, concentrations, labels
