"""Apply a fitted model bundle to new spectra."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import load_models


def predict(spectra: pd.DataFrame, models: dict | str | Path) -> pd.DataFrame:
    """Predict concentrations (mol/L) and, if the bundle has one, the Fe class.

    Parameters
    ----------
    spectra : rows = samples, columns = wavelengths (nm), exactly as in calibration.
    models : a bundle from :func:`ndfeb_uvvis.fit_models` or a path to a saved bundle.
    """
    m = models if isinstance(models, dict) else load_models(models)
    expected = m["wavelengths"]
    got = spectra.columns.to_numpy(dtype=np.float64)
    if got.shape != expected.shape or not np.allclose(got, expected):
        raise ValueError(
            f"Spectra must have the calibration wavelengths "
            f"({expected[0]:g}-{expected[-1]:g} nm, {expected.size} columns); got {got.size} columns"
        )
    X = spectra.to_numpy(dtype=np.float64)

    results = pd.DataFrame(index=spectra.index)
    X_pre = m["preprocessing"].transform(X)  # transform only: reuses calibration mean centring
    for target, r in m["regressors"].items():
        results[f"{target} (mol/L)"] = np.ravel(r["model"].predict(X_pre[:, r["mask"]]))

    if "fe_model" in m:
        score = np.ravel(m["fe_model"].predict(m["fe_preprocessing"].transform(X)))
        results["Fe score"] = score
        results["Fe class"] = m["fe_classes"][(score > m["fe_threshold"]).astype(int)]
    return results
