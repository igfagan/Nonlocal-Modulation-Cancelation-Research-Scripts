"""
beta calibration + matching simulation for an AOPM and an EOM (v2)

Key correction from v1: V_pi is NOT known in advance. In the lab, beta
is measured directly and independently for each device (e.g. via
RF/optical sideband-to-carrier ratios on a spectrum analyzer, fit to
Bessel functions J_n(beta)), as a function of the RF drive power set
at the source. V_pi for each device is then *extracted* from that
device's own measured calibration curve via a fit -- beta scales as
sqrt(P_RF), so beta vs sqrt(P_delivered) is a straight line through
the origin whose slope is pi/V_pi. No cross-device information is
needed to get V_pi; each device is calibrated on its own.

The cross-device "beta matching" step is separate and comes *after*:
once each device has its own measured (fitted) calibration curve, you
pick a common beta that both devices can reach and read off the two
different P_RF values -- one per arm (signal/idler) -- that produce it
in the real experiment.

Pipeline simulated here:
  1. Generate synthetic noisy "measured" beta vs P_RF data for each
     device from a HIDDEN true V_pi (stand-in for a real sideband
     measurement). The true V_pi is only used to generate the fake
     data -- everything downstream pretends not to know it, exactly
     as a real experimenter wouldn't.
  2. Fit each device's own data to recover V_pi (and compare the
     recovered value to the hidden truth as a sanity check on the
     simulation, not something available in a real experiment).
  3. Use the FITTED calibration curves -- not the hidden truth -- to
     do the beta matching: find P_RF_EOM and P_RF_AOPM that produce a
     common beta.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

R = 50.0          # ohm, RF load impedance
F_RF = 3.64e9      # Hz, drive frequency (labeling only)

# ---------------------------------------------------------------------------
# Hidden ground truth (HYPOTHETICAL, stand-in for the real, physical V_pi
# values). In a real experiment these are UNKNOWN until you run the
# calibration below and fit for them -- they are only used here to
# generate synthetic "measured" data.
#
# Insertion/cable loss, by contrast, is something you'd typically measure
# independently and simply with an RF power meter at the device input, so
# it's treated as known/given here rather than something to fit for.
# ---------------------------------------------------------------------------
TRUE_DEVICES = {
    "EOM":  {"V_pi_true": 4.5, "insertion_loss_dB": 0.8, "color": "#1f77b4"},
    "AOPM": {"V_pi_true": 6.1, "insertion_loss_dB": 1.5, "color": "#d62728"},
}

BETA_MEAS_NOISE_REL = 0.03   # 3% relative noise on each beta measurement
BETA_MEAS_NOISE_ABS = 0.01   # rad, small additive noise floor

N_CAL_POINTS = 14
P_RF_CAL_DBM = np.linspace(-6.0, 22.0, N_CAL_POINTS)   # discrete cal sweep,
                                                        # as you'd actually
                                                        # step a signal
                                                        # generator in the lab


def dbm_to_watts(p_dbm):
    return 1e-3 * 10 ** (p_dbm / 10.0)


def true_beta(p_rf_dbm, v_pi, insertion_loss_dB, R=R):
    """Ground-truth beta -- used only to fabricate synthetic 'measured' data."""
    p_delivered_w = dbm_to_watts(p_rf_dbm - insertion_loss_dB)
    v_pk = np.sqrt(2.0 * R * p_delivered_w)
    return np.pi * v_pk / v_pi


# ---------------------------------------------------------------------------
# Step 1: synthesize noisy "measured" calibration data per device.
# Stands in for a real per-point sideband-ratio (Bessel fit) measurement.
# ---------------------------------------------------------------------------
measured = {}
for name, dev in TRUE_DEVICES.items():
    beta_true = true_beta(P_RF_CAL_DBM, dev["V_pi_true"], dev["insertion_loss_dB"])
    noise = (rng.normal(0.0, BETA_MEAS_NOISE_ABS, size=beta_true.shape)
             + beta_true * rng.normal(0.0, BETA_MEAS_NOISE_REL, size=beta_true.shape))
    beta_meas = np.clip(beta_true + noise, 0.0, None)
    measured[name] = {"P_RF_dBm": P_RF_CAL_DBM.copy(), "beta": beta_meas}


# ---------------------------------------------------------------------------
# Step 2: fit each device's OWN data to recover V_pi.
# Model: beta = k * sqrt(P_delivered_W), where k = pi*sqrt(2R)/V_pi.
# This is a line through the origin in (x, beta) with x = sqrt(P_delivered_W)
# -- an ordinary least-squares fit forced through zero (beta must vanish at
# zero drive).
# ---------------------------------------------------------------------------
def fit_v_pi(p_rf_dbm, beta_meas, insertion_loss_dB, R=R):
    p_delivered_w = dbm_to_watts(p_rf_dbm - insertion_loss_dB)
    x = np.sqrt(p_delivered_w)
    k = np.sum(x * beta_meas) / np.sum(x ** 2)          # OLS through origin
    resid = beta_meas - k * x
    dof = len(x) - 1
    sigma_k = np.sqrt(np.sum(resid ** 2) / dof / np.sum(x ** 2)) if dof > 0 else np.nan
    v_pi = np.pi * np.sqrt(2.0 * R) / k
    sigma_v_pi = v_pi * (sigma_k / k)                   # linear error propagation
    return v_pi, sigma_v_pi, k


fitted = {}
for name, dev in TRUE_DEVICES.items():
    v_pi_fit, sigma_v_pi, k_fit = fit_v_pi(
        measured[name]["P_RF_dBm"], measured[name]["beta"], dev["insertion_loss_dB"]
    )
    fitted[name] = {
        "V_pi": v_pi_fit, "sigma_V_pi": sigma_v_pi, "k": k_fit,
        "insertion_loss_dB": dev["insertion_loss_dB"], "color": dev["color"],
    }
    print(f"{name}: fitted V_pi = {v_pi_fit:.3f} +/- {sigma_v_pi:.3f} V   "
          f"[hidden truth = {dev['V_pi_true']:.3f} V, recovery check only]")


def beta_from_prf_dbm_fitted(p_rf_dbm, name, R=R):
    dev = fitted[name]
    p_delivered_w = dbm_to_watts(p_rf_dbm - dev["insertion_loss_dB"])
    return dev["k"] * np.sqrt(p_delivered_w)


def prf_dbm_from_beta_fitted(beta_target, name, R=R):
    dev = fitted[name]
    p_delivered_w = (beta_target / dev["k"]) ** 2
    p_delivered_dbm = 10.0 * np.log10(p_delivered_w / 1e-3)
    return p_delivered_dbm + dev["insertion_loss_dB"]


# ---------------------------------------------------------------------------
# Step 3: beta matching using the FITTED (measured) calibration curves --
# this is the only step that looks at both devices at once, and it happens
# after each V_pi is already known from its own independent fit.
# ---------------------------------------------------------------------------
p_rf_plot_dbm = np.linspace(P_RF_CAL_DBM.min(), P_RF_CAL_DBM.max(), 400)
fitted_curves = {name: beta_from_prf_dbm_fitted(p_rf_plot_dbm, name) for name in TRUE_DEVICES}
max_beta_each = {name: fitted_curves[name].max() for name in TRUE_DEVICES}

beta_target = 0.6 * min(max_beta_each.values())   # representative operating point

match_results = {name: prf_dbm_from_beta_fitted(beta_target, name) for name in TRUE_DEVICES}

print(f"\nTarget beta = {beta_target:.4f} rad (common operating point)\n")
for name in TRUE_DEVICES:
    print(f"  {name}: P_RF = {match_results[name]:.2f} dBm "
          f"({dbm_to_watts(match_results[name]) * 1e3:.2f} mW)")
delta_dbm = match_results["AOPM"] - match_results["EOM"]
print(f"\nP_RF(AOPM) - P_RF(EOM) = {delta_dbm:+.2f} dB "
      f"(different, as expected, since the fitted V_pi differs)")

# ---------------------------------------------------------------------------
# Plot: raw noisy "measurements" + fitted calibration curve per device,
# plus the matched operating point.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

for name, dev in fitted.items():
    c = dev["color"]
    ax.plot(measured[name]["P_RF_dBm"], measured[name]["beta"], "o",
             color=c, ms=5, alpha=0.6, label=f"{name} measured points")
    ax.plot(p_rf_plot_dbm, fitted_curves[name], "-", color=c, lw=2,
             label=f"{name} fit (V$_\\pi$ = {dev['V_pi']:.2f} $\\pm$ "
                    f"{dev['sigma_V_pi']:.2f} V)")
    ax.plot(match_results[name], beta_target, "o", color=c, ms=9,
             mec="k", zorder=5)

ax.axhline(beta_target, color="gray", ls="--", lw=1)
ax.annotate(f"$\\beta$ = {beta_target:.3f} rad",
            xy=(P_RF_CAL_DBM[0], beta_target),
            xytext=(4, 4), textcoords="offset points", fontsize=9, color="gray")

label_offsets = {"EOM": (-8, -18), "AOPM": (8, 14)}
label_ha = {"EOM": "right", "AOPM": "left"}
for name in TRUE_DEVICES:
    ax.annotate(f"{match_results[name]:.1f} dBm",
                xy=(match_results[name], beta_target),
                xytext=label_offsets[name], textcoords="offset points",
                ha=label_ha[name], fontsize=9, color=fitted[name]["color"])

ax.set_xlabel("RF drive power at source, $P_{RF}$ (dBm)")
ax.set_ylabel(r"Modulation index, $\beta$ (rad)")
ax.set_title(f"Simulated $\\beta$ calibration + matching @ {F_RF/1e9:.2f} GHz\n"
             f"(V$_\\pi$ recovered per-device from noisy calibration data)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "beta_vs_prf_calibration.png"
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
