"""Figures of merit for validating the regression and classification models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_report(y_true, y_pred) -> dict[str, float]:
    """RMSEP, bias, SEP (bias-corrected standard error), R² and mean relative error.

    Mean relative error is computed only over samples with a non-zero reference value.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_pred - y_true
    bias = err.mean()
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    nz = y_true != 0
    return {
        "n": int(y_true.size),
        "RMSEP": float(np.sqrt(np.mean(err**2))),
        "Bias": float(bias),
        "SEP": float(np.sqrt(np.sum((err - bias) ** 2) / (y_true.size - 1))) if y_true.size > 1 else np.nan,
        "R2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else np.nan,
        "Mean rel. error (%)": float(100 * np.mean(np.abs(err[nz]) / y_true[nz])) if nz.any() else np.nan,
    }


def validation_table(reference: pd.DataFrame, predictions: pd.DataFrame, targets) -> pd.DataFrame:
    """One row of :func:`regression_report` per target.

    ``predictions`` uses the column names produced by :func:`ndfeb_uvvis.predict`
    (``"<target> (mol/L)"``).
    """
    rows = {t: regression_report(reference[t], predictions[f"{t} (mol/L)"]) for t in targets}
    return pd.DataFrame(rows).T
