"""Train, cross-validate, save and load the Nd/Pr PLS and Fe PLS-DA models.

Preprocessing (applied to every model):

1. range cut to 400-902 nm
2. linear baseline correction
3. Savitzky-Golay first derivative (window 7, 2nd-order polynomial)
4. mean centring

This reproduces the settings of the original MATLAB PLS_Toolbox models, re-implemented
with open libraries (scikit-learn and chemotools). No Eigenvector code is used.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from chemotools.baseline import LinearCorrection
from chemotools.derivative import SavitzkyGolay
from chemotools.feature_selection import RangeCut
from chemotools.regression import PLSRegression
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

#: Default number of latent variables per target.
DEFAULT_COMPONENTS = {"Nd (3+)": 1, "Pr (3+)": 3}
DEFAULT_FE_COMPONENTS = 2


def _ranges(*spans: tuple[float, float], step: float = 2.0) -> np.ndarray:
    return np.concatenate([np.arange(lo, hi + step / 2, step) for lo, hi in spans])


#: Wavelength selection (nm) used in the published models, on the 2 nm instrument grid.
#: Nd: 125 wavelengths around the 520, 575, 740, 794 and 865 nm bands.
#: Pr: 23 wavelengths around the 444, 468, 482 and 590 nm bands.
PUBLISHED_WAVELENGTHS: dict[str, np.ndarray] = {
    "Nd (3+)": _ranges(
        (496, 510), (514, 538), (560, 598), (672, 676), (684, 690),
        (722, 740), (744, 766), (774, 832), (842, 882), (890, 896),
    ),
    "Pr (3+)": _ranges((436, 454), (466, 470), (480, 488), (596, 604)),
}


def wavelengths_of(spectra: pd.DataFrame) -> np.ndarray:
    """Wavelength axis (nm) from the DataFrame column headers."""
    return spectra.columns.to_numpy(dtype=np.float64)


def build_preprocessing(
    wavelengths: np.ndarray,
    start: float = 400,
    end: float = 902,
    window_length: int = 7,
    polyorder: int = 2,
) -> Pipeline:
    """Unfitted preprocessing pipeline: range cut, baseline, SG 1st derivative, mean centring."""
    return make_pipeline(
        RangeCut(start=start, end=end, x_axis=np.asarray(wavelengths, dtype=np.float64)),
        LinearCorrection(),
        SavitzkyGolay(window_length=window_length, polyorder=polyorder, deriv=1),
        StandardScaler(with_std=False),  # mean centring
    )


class VenetianBlinds(BaseCrossValidator):
    """Venetian-blinds cross-validation, as used by PLS_Toolbox.

    Sample ``i`` goes to fold ``i % n_splits``, so every fold spans the whole
    sample order. scikit-learn's plain ``KFold`` uses contiguous blocks instead,
    which gives different (often more pessimistic) errors on ordered data.
    """

    def __init__(self, n_splits: int = 10):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None) -> Iterator[np.ndarray]:
        n = len(X)
        idx = np.arange(n)
        for k in range(self.n_splits):
            yield idx[idx % self.n_splits == k]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def cross_validate_components(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int = 10,
    cv: BaseCrossValidator | None = None,
) -> pd.DataFrame:
    """RMSECV for 1..max_components latent variables on already-preprocessed X.

    Mean centring inside ``X`` is fitted on all samples, as in PLS_Toolbox's
    default cross-validation of preprocessed data.
    """
    cv = cv or VenetianBlinds(10)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    max_components = min(max_components, X.shape[1], len(y) - len(y) // cv.get_n_splits() - 1)
    rows = []
    for a in range(1, max_components + 1):
        pred = np.empty_like(y)
        for train, test in cv.split(X, y):
            m = PLSRegression(n_components=a, scale=False).fit(X[train], y[train])
            pred[test] = np.ravel(m.predict(X[test]))
        rows.append({"n_components": a, "RMSECV": float(np.sqrt(np.mean((pred - y) ** 2)))})
    return pd.DataFrame(rows).set_index("n_components")


def fit_models(
    spectra: pd.DataFrame,
    concentrations: pd.DataFrame,
    labels: pd.Series | None = None,
    *,
    n_components: dict[str, int] | None = None,
    selected_wavelengths: dict[str, np.ndarray] | None = None,
    fe_components: int = DEFAULT_FE_COMPONENTS,
) -> dict:
    """Fit the PLS regressors (and optionally the Fe PLS-DA classifier).

    Parameters
    ----------
    spectra : rows = samples, columns = wavelengths (nm).
    concentrations : one column per target, e.g. ``"Nd (3+)"`` (mol/L).
    labels : ``"High"`` / ``"Low"`` Fe class per sample; omit to skip PLS-DA.
    n_components : latent variables per target; defaults to :data:`DEFAULT_COMPONENTS`.
    selected_wavelengths : optional per-target wavelength subset (nm) after range cut.
        Targets not listed use every wavelength in the range.

    Returns a model bundle for :func:`save_models` and :func:`ndfeb_uvvis.predict`.
    """
    if len(spectra) != len(concentrations):
        raise ValueError(f"Sample count mismatch: {len(spectra)} spectra vs {len(concentrations)} concentrations")
    n_components = n_components or DEFAULT_COMPONENTS
    selected_wavelengths = selected_wavelengths or {}

    wl = wavelengths_of(spectra)
    X = spectra.to_numpy(dtype=np.float64)
    preprocessing = build_preprocessing(wl)
    X_pre = preprocessing.fit_transform(X)
    wl_cut = preprocessing.named_steps["rangecut"].x_axis_

    regressors = {}
    for target, a in n_components.items():
        sel = selected_wavelengths.get(target)
        mask = np.ones(wl_cut.size, bool) if sel is None else np.isin(wl_cut, sel)
        if not mask.any():
            raise ValueError(f"No selected wavelengths for {target} fall inside the range cut")
        model = PLSRegression(n_components=a, scale=False)
        model.fit(X_pre[:, mask], concentrations[target].to_numpy(dtype=float))
        regressors[target] = {"mask": mask, "model": model}

    bundle = {
        "wavelengths": wl,
        "preprocessing": preprocessing,
        "regressors": regressors,
        "versions": _versions(),
    }

    if labels is not None:
        if len(labels) != len(spectra):
            raise ValueError("labels must have one entry per spectrum")
        fe_preprocessing = clone(preprocessing)
        X_fe = fe_preprocessing.fit_transform(X)
        encoder = LabelEncoder().fit(np.asarray(labels))
        fe_model = PLSRegression(n_components=fe_components, scale=False)
        fe_model.fit(X_fe, encoder.transform(np.asarray(labels)).astype(float))
        bundle.update(
            fe_preprocessing=fe_preprocessing,
            fe_model=fe_model,
            fe_classes=encoder.classes_,
            fe_threshold=0.5,
        )
    return bundle


def save_models(bundle: dict, path: str | Path) -> Path:
    """Save a model bundle with joblib. Reload with the same library versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_models(path: str | Path) -> dict:
    """Load a bundle saved by :func:`save_models`, warning on library version changes."""
    bundle = joblib.load(Path(path))
    saved, now = bundle.get("versions", {}), _versions()
    changed = {k: (saved[k], now.get(k)) for k in saved if saved[k] != now.get(k)}
    if changed:
        import warnings

        warnings.warn(
            f"Models were saved with different library versions {changed}; "
            "retrain before relying on the predictions.",
            stacklevel=2,
        )
    return bundle


def _versions() -> dict[str, str]:
    out = {}
    for pkg in ("scikit-learn", "chemotools", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out
