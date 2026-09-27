"""End-to-end validation on synthetic data.

    python examples/validate_synthetic.py

1. Simulates independent calibration (n = 250) and test (n = 60) sets.
2. Applies the published wavelength selection and checks the latent variables
   against venetian-blinds RMSECV.
3. Fits the Nd/Pr PLS and Fe PLS-DA models and predicts the test set.
4. Writes a figures-of-merit table and a figure to docs/.

Everything here uses simulated spectra; see src/ndfeb_uvvis/simulate.py.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ndfeb_uvvis as nu

OUT = Path(__file__).resolve().parents[1] / "docs"
TARGETS = ["Nd (3+)", "Pr (3+)"]
COLOURS = {"Nd (3+)": "#2a78d6", "Pr (3+)": "#eb6834"}
INK, MUTED = "#0b0b0b", "#52514e"

# 1. Data
cal_spectra, cal_conc, cal_labels = nu.simulate(250, seed=0)
test_spectra, test_conc, test_labels = nu.simulate(60, seed=1)

# 2. Wavelength selection and latent variables
# The wavelength selection from the published models (nu.PUBLISHED_WAVELENGTHS):
#   Nd: 125 wavelengths around the 520, 575, 740, 794 and 865 nm bands.
#   Pr: 23 wavelengths around the 444, 468, 482 and 590 nm bands.
windows = nu.PUBLISHED_WAVELENGTHS
pre = nu.build_preprocessing(nu.model.wavelengths_of(cal_spectra))
X_cal = pre.fit_transform(cal_spectra.to_numpy())
wl_cut = pre.named_steps["rangecut"].x_axis_
cv = {
    t: nu.cross_validate_components(X_cal[:, np.isin(wl_cut, windows[t])], cal_conc[t], max_components=8)["RMSECV"]
    for t in TARGETS
}
print(pd.DataFrame(cv).round(5).to_string())
# The latent variables of the published models: one for Nd, whose bands dominate its selected
# wavelengths, and three for Pr. In the simulation the second Pr LV mainly captures the Fe(III)
# tail in 436-454 nm, and the third smaller overlaps (the broad Co(II) band) and band shifts.
# Beyond three, RMSECV falls only slightly, mainly by modelling the simulated wavelength shifts.
n_components = nu.model.DEFAULT_COMPONENTS
print("Latent variables:", n_components)

# 3. Fit and predict
models = nu.fit_models(cal_spectra, cal_conc, cal_labels, n_components=n_components,
                       selected_wavelengths=windows)
pred = nu.predict(test_spectra, models)
table = nu.validation_table(test_conc, pred, TARGETS)
table.insert(0, "LVs", [n_components[t] for t in TARGETS])
table.insert(1, "RMSECV", [cv[t].loc[n_components[t]] for t in TARGETS])
fe_acc = float((pred["Fe class"].to_numpy() == test_labels.to_numpy()).mean())
print(table.round(4).to_string())
print(f"Fe PLS-DA accuracy on test set: {fe_acc:.1%}")

OUT.mkdir(exist_ok=True)
fmt = table.copy().astype(object)
fmt["LVs"] = table["LVs"].astype(int).astype(str)
fmt["n"] = table["n"].astype(int).astype(str)
for col in ["RMSECV", "RMSEP", "Bias", "SEP", "R2"]:
    fmt[col] = table[col].map("{:.4f}".format)
fmt["Mean rel. error (%)"] = table["Mean rel. error (%)"].map("{:.1f}".format)
(OUT / "synthetic_validation.md").write_text(
    "Concentrations and errors in mol/L.\n\n" + fmt.to_markdown(stralign="right", disable_numparse=True) + f"\n\nFe(III) PLS-DA test-set accuracy: {fe_acc:.1%} (n = {len(pred)})\n",
    encoding="utf-8",
)

# 4. Figure: RMSECV curves | predicted vs reference | Fe PLS-DA scores
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
for t in TARGETS:
    ax.plot(cv[t].index, cv[t].values * 1000, lw=2, color=COLOURS[t], marker="o", ms=5, label=t)
    ax.plot(n_components[t], cv[t].loc[n_components[t]] * 1000, "o", ms=11, mfc="none", mec=COLOURS[t], mew=2)
ax.set(xlabel="Latent variables", ylabel="RMSECV (mmol/L)", yscale="log",
       title="RMSECV on selected wavelengths (venetian blinds)")
ax.set_xticks(range(1, 9))
ax.legend(frameon=False)

ax = axes[1]
lim = max(test_conc[TARGETS].max().max(), pred[[f"{t} (mol/L)" for t in TARGETS]].max().max()) * 1.05
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="1:1")
for t in TARGETS:
    ax.scatter(test_conc[t], pred[f"{t} (mol/L)"], s=36, color=COLOURS[t], edgecolor="white", lw=1,
               label=f"{t}  RMSEP {table.loc[t, 'RMSEP'] * 1000:.1f} mmol/L")
ax.set(xlim=(0, lim), ylim=(0, lim), aspect="equal", xlabel="Reference (mol/L)", ylabel="Predicted (mol/L)",
       title=f"Independent test set (n = {len(pred)})")
ax.legend(frameon=False, loc="upper left")

ax = axes[2]
for cls, colour in [("High", COLOURS["Pr (3+)"]), ("Low", COLOURS["Nd (3+)"])]:
    sel = test_labels.to_numpy() == cls
    ax.scatter(test_conc["Fe (3+)"][sel], pred["Fe score"][sel], s=36, color=colour, edgecolor="white", lw=1,
               label=f"True {cls} Fe")
ax.axhline(models["fe_threshold"], color=MUTED, lw=1, ls="--", label="Threshold 0.5")
ax.set(xlabel="Fe(III) reference (mol/L)", ylabel="PLS-DA score (0 = High, 1 = Low)",
       title=f"Fe(III) PLS-DA, accuracy {fe_acc:.0%}")
ax.legend(frameon=False)

for ax in axes:
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "synthetic_validation.png", dpi=150)
print(f"Wrote {OUT / 'synthetic_validation.md'} and {OUT / 'synthetic_validation.png'}")
