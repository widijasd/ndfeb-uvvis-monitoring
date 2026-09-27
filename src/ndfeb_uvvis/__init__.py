"""UV-Vis chemometric models for monitoring hydrometallurgical recycling of NdFeB magnets.

    import ndfeb_uvvis as nu

    spectra, conc, labels = nu.simulate(250, seed=0)       # synthetic stand-in data
    models = nu.fit_models(spectra, conc, labels)
    nu.predict(new_spectra, models)
"""
from .metrics import regression_report, validation_table
from .model import (
    PUBLISHED_WAVELENGTHS,
    VenetianBlinds,
    build_preprocessing,
    cross_validate_components,
    fit_models,
    load_models,
    save_models,
)
from .predict import predict
from .simulate import Design, Effects, simulate

__all__ = [
    "Design",
    "PUBLISHED_WAVELENGTHS",
    "Effects",
    "VenetianBlinds",
    "build_preprocessing",
    "cross_validate_components",
    "fit_models",
    "load_models",
    "predict",
    "regression_report",
    "save_models",
    "simulate",
    "validation_table",
]
__version__ = "0.1.0"
