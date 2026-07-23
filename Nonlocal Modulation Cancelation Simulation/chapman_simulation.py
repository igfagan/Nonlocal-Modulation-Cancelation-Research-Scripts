"""
Simulation of Figures 5a, 5b, 5c from:
  Chapman et al., "Quantum nonlocal modulation cancelation with distributed clocks,"
  Optica Quantum 3, 45 (2025).

Figure 5a: Unmodulated JSI (both EOPMs off)
Figure 5b: In-phase modulation (phi_A = phi_B, Delta = 1.42 rad)
Figure 5c: Out-of-phase modulation (phi_A - phi_B = pi, Delta = 1.42 rad)
Figure 5d: Drifting (failed sync) — incoherent average over all phases

Theory: Eq. (6) of the paper.
    R_jk = sum_p [ overlap(p) ]
             x | sum_m J_m(Delta_A) J_{m-j+k-p}(Delta_B) exp(-im*(phi_A-phi_B+pi)) |^2

    Bin convention (matching paper Fig. 5):
        Signal bin j : frequency  omega_0 + j*Omega       (j = 1...9, increases left->right)
        Idler  bin k : frequency  omega_0 - k*Omega       (k = 1...9, increases top->bottom)
        Idler frequency DECREASES as k increases, so bin 1 = highest idler freq.
        Energy conservation: j = k  (MAIN DIAGONAL, displayed as "\" with origin='upper').

    overlap(p) = overlap(0) * exp(-2*ln2 * p^2 * Omega^2 / Gamma^2)  [Gaussian filters]

Phase matching envelope (textbook Eq. 6.27):
    E(j) = sinc^2( |k''| * ((j-5)*Omega)^2 * L/2 )
    For 2 cm PPLN at 1560 nm: E ~ 1 across all 9 bins (BW >> passband).
    Set bw_scale < 1 to artificially narrow the BW for illustration.

Noise model (textbook Sec. 6.2.2):
    (a) Detector efficiency — beam-splitter loss model (Eq. 6.46/6.52):
        True coincidences are Poisson with mean scaled by eta_A * eta_B.
    (b) Thermal accidentals — multi-pair SPDC statistics (Eq. 6.54-6.56):
        Accidentals follow a negative binomial distribution.
        variance = mean + mean^2 / M_modes

Output files:
    fig5_jsi_all.png     — 2x2 grid: 5a, 5b, 5c, drifting JSIs
    fig5_slices.png      — 3x2 grid: JSI + marginal slice for 5a, 5b, 5c
    fig5_phase_sweep.png — R(5,5) vs phi_diff sweeping 0 -> 2pi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
   # Bessel functions of the first kind

# ============================================================
# Physical parameters
# ============================================================
Omega      = 19e9     # Hz   — bin spacing / RF frequency
Gamma      = 30e9     # Hz   — Gaussian filter FWHM
Delta      = 1.42     # rad  — modulation depth
N_bins     = 9        # bins per axis (labeled 1 ... 9)
pass_lo    = 2        # first bin inside 140 GHz passband
pass_hi    = 8        # last  bin inside 140 GHz passband
center_bin = 5        # bin at degeneracy (omega_0)

M_max = 12    # Bessel sum truncation  (converged for Delta = 1.42)
P_max = 4     # overlap sum truncation (|p|>2 contributes < 0.01%)

# Pre-build index arrays once — reused in every compute_Rjk call
_m_arr = np.arange(-M_max, M_max + 1)   # shape (2*M_max+1,)
_p_arr = np.arange(-P_max, P_max + 1)   # shape (2*P_max+1,)

# Phase matching (PPLN, textbook Eq. 6.27)
L_PPLN   = 0.02    # m      — waveguide length (2 cm, from paper)
k2       = 1e-22   # s^2/m — |GVD| at degeneracy (~100 ps^2/km)
bw_scale = 1.0     # 1.0 = physical; < 1 narrows BW for illustration

# Scaling  (CAR~25, off-diag accidentals~100, exterior~6 — from paper)
c0_inside  = 100.0
c0_outside = 6.0
c1         = 25.0 * c0_inside    # diagonal peak ~ 2600 counts

# ------------------------------------------------------------------
# Detector + noise model  (textbook Sec. 6.2.2)
# ------------------------------------------------------------------
eta_A   = 0.85    # signal-arm SNSPD efficiency (beam-splitter loss model)
eta_B   = 0.85    # idler-arm  SNSPD efficiency
M_modes = 10      # SPDC spectral modes (thermal noise: var = mean + mean^2/M_modes)

# ------------------------------------------------------------------
# Display settings
# ------------------------------------------------------------------
slice_signal_bin = 5      # which signal bin to show in the marginal slice plots
shared_scale     = False   # True  = all JSI panels share one colour scale
                          # False = each panel independently auto-scaled
N_drift          = 36     # phase steps for the incoherent (drifting) average

# ------------------------------------------------------------------
# Experimental imperfections (applied only to out-of-phase case)
# ------------------------------------------------------------------
phase_error      = 0.04   # rad — residual clock phase-lock error (δφ)
                          # real clocks never hit exactly π; 0.04 rad ≈ 2.3°
filter_sigma_bins = 1.0   # soft Gaussian passband rolloff (in bin units)
                          # bins 1 & 9 are 1 bin outside edge → ~61% transmission

# Random seed — change to None for a fresh noise realisation each run
rng = np.random.default_rng(seed=42)

# ============================================================
# Helper functions
# ============================================================

def filter_weight(bin_idx):
    """
    Soft Gaussian passband rolloff for bin bin_idx (1-indexed).
    Returns 1.0 inside the hard passband (pass_lo to pass_hi).
    Decays as a Gaussian beyond the passband edges, with sigma = filter_sigma_bins.
    Bins 1 and 9 are 1 bin outside the edge, so receive exp(-0.5/sigma^2) weight.
    """
    dist = max(0.0, pass_lo - bin_idx, bin_idx - pass_hi)   # 0 inside passband
    return np.exp(-0.5 * (dist / filter_sigma_bins)**2)


def overlap_ratio(p):
    """
    overlap(p) / overlap(0) for Gaussian filters — analytical convolution result.
    Works with scalar p or a numpy array.
    """
    return np.exp(-2.0 * np.log(2.0) * p**2 * (Omega / Gamma)**2)


def phase_match_envelope(j):
    """
    Sinc^2 phase-matching envelope at signal bin j (1-indexed).
    Detuning from degeneracy: delta_omega = (j - center_bin) * Omega.
    """
    delta_omega = (j - center_bin) * Omega
    beta = (k2 / bw_scale) * delta_omega**2 * L_PPLN / 2.0
    if beta == 0.0:
        return 1.0
    return (np.sin(beta) / beta)**2


def compute_Rjk(j, k, Delta_A, Delta_B, phi_diff):
    """
    Coincidence rate R_jk from Eq. (6) — fully vectorized over m and p.

    Both Python for-loops replaced with NumPy array operations:
      - bj_A  : Bessel J_m(Delta_A)          shape (2M+1,)
      - phases: exp(-im*(phi_diff+pi))        shape (2M+1,)
      - bj_B  : J_{m-j+k-p}(Delta_B)         shape (2M+1, 2P+1)
      - inner_per_p = sum over m              shape (2P+1,)
      - R = sum over p of ov * |inner|^2      scalar
    """
    bj_A   = jv(_m_arr, Delta_A)                                 # (2M+1,)
    phases = np.exp(-1j * _m_arr * (phi_diff + np.pi))           # (2M+1,)
    ov_arr = overlap_ratio(_p_arr)                               # (2P+1,)

    # Idler Bessel indices: m - j + k - p  for all (m, p) combinations
    indices = _m_arr[:, None] - j + k - _p_arr[None, :]         # (2M+1, 2P+1)
    bj_B    = jv(indices, Delta_B)                               # (2M+1, 2P+1)

    # Sum over m for each p
    inner_per_p = np.sum(
        bj_A[:, None] * bj_B * phases[:, None], axis=0          # (2P+1,)
    )

    # Sum over p, skipping negligible overlaps
    mask = ov_arr > 1e-14
    return float(np.sum(ov_arr[mask] * np.abs(inner_per_p[mask])**2))


def build_JSI(Delta_A, Delta_B, phi_diff, phi_error=0.0):
    """
    Build the 9x9 JSI matrix of expected coincidence counts.
    C[ji, ki]: ji = j-1 (signal, 0-indexed), ki = k-1 (idler, 0-indexed).

    All bins receive signal weighted by the soft Gaussian filter rolloff.
    Bins inside the passband (2-8) get full weight; bins 1 and 9 get partial
    transmission from the filter edge tails.

    phi_error : add a small phase offset to phi_diff (residual clock
                error δφ), preventing perfect destructive interference.
    """
    R_norm   = compute_Rjk(center_bin, center_bin, 0.0, 0.0, 0.0)
    phi_eff  = phi_diff + phi_error
    C = np.zeros((N_bins, N_bins))
    for ji in range(N_bins):
        j = ji + 1
        pm_env = phase_match_envelope(j)
        w_j    = filter_weight(j)
        for ki in range(N_bins):
            k = ki + 1
            w_k = filter_weight(k)
            R = compute_Rjk(j, k, Delta_A, Delta_B, phi_eff) / R_norm
            R *= pm_env * w_j * w_k
            idler_in_pass = (pass_lo <= k <= pass_hi)
            c0 = c0_inside if idler_in_pass else c0_outside
            C[ji, ki] = c1 * R + c0
    return C


def build_JSI_drifting(Delta_A, Delta_B):
    """
    Simulate FAILED RF synchronisation: average the JSI incoherently over
    N_drift uniformly spaced phase offsets spanning 0 -> 2*pi.

    Physical meaning (paper Fig. 4a-ii):
        When the two RF clocks drift by more than one RF period during the
        measurement, all relative phases are equally likely.  The measured
        JSI becomes an incoherent average — in-phase and out-of-phase
        modulation produce identical results, and the nonlocal cancellation
        effect is completely washed out.
    """
    JSI_avg = np.zeros((N_bins, N_bins))
    phases  = np.linspace(0, 2 * np.pi, N_drift, endpoint=False)
    for phi in phases:
        JSI_avg += build_JSI(Delta_A, Delta_B, phi_diff=phi)
    return JSI_avg / N_drift


# ============================================================
# Sanity checks: print phase matching envelope
# ============================================================
print("Phase-matching sinc² envelope (bw_scale={:.2f}):".format(bw_scale))
for b in range(1, N_bins + 1):
    print("  bin {:d}  ({:+.0f} GHz): E = {:.6f}".format(
        b, (b - center_bin) * Omega / 1e9, phase_match_envelope(b)))
print()

# ============================================================
# Compute JSI matrices for all four scenarios
# ============================================================
print("Computing Fig 5a (unmodulated)...")
JSI_a = build_JSI(0.0,   0.0,   0.0)

print("Computing Fig 5b (in-phase, Delta=1.42 rad)...")
JSI_b = build_JSI(Delta, Delta,  0.0)

print("Computing Fig 5c (out-of-phase, Delta=1.42 rad)...")
JSI_c = build_JSI(Delta, Delta, -np.pi, phi_error=phase_error)

print("Computing Fig 5d (drifting / failed sync, {:d} phase steps)...".format(N_drift))
JSI_d = build_JSI_drifting(Delta, Delta)

print("Done.\n")

# ============================================================
# Metrics: CAR and JSI fidelity
# ============================================================
centre_idx = center_bin - 1   # 0-indexed

# Coincidence-to-accidental ratio at the centre diagonal bin
car = (JSI_a[centre_idx, centre_idx] - c0_inside) / c0_inside
print("CAR at bin ({0:d},{0:d}): {1:.1f}".format(center_bin, car))

# JSI fidelity: how well does out-of-phase (5c) recover the unmodulated (5a)?
# Computed as normalised dot product (cosine similarity) of the two matrices.
fidelity = (np.sum(JSI_a * JSI_c) /
            np.sqrt(np.sum(JSI_a**2) * np.sum(JSI_c**2)))
print("JSI fidelity  5c vs 5a : {:.6f}  (1.0 = perfect recovery)".format(fidelity))

# How different is the drifting case from the unmodulated case?
fidelity_drift = (np.sum(JSI_a * JSI_d) /
                  np.sqrt(np.sum(JSI_a**2) * np.sum(JSI_d**2)))
print("JSI fidelity  5d vs 5a : {:.6f}  (shows degradation from drift)\n".format(fidelity_drift))

# ============================================================
# Plotting helpers
# ============================================================
bin_labels = [str(b) for b in range(1, N_bins + 1)]
bin_ticks  = list(range(N_bins))
cmap       = 'viridis'

# Shared colour scale driven by the in-phase case (broadest spread)
vmax_shared = np.max(JSI_b)


def plot_JSI(ax, JSI, title, vmin=0, vmax=None):
    """
    Plot a JSI heatmap.  origin='upper' puts idler bin 1 at the top,
    matching the paper.  JSI.T so columns=signal (x), rows=idler (y).
    """
    if vmax is None:
        vmax = np.max(JSI)
    im = ax.imshow(
        JSI.T,
        origin='upper',
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        aspect='equal',
        interpolation='nearest',
    )
    ax.set_xticks(bin_ticks);  ax.set_xticklabels(bin_labels)
    ax.set_yticks(bin_ticks);  ax.set_yticklabels(bin_labels)
    ax.set_xlabel("Signal Bin", fontsize=11)
    ax.set_ylabel("Idler Bin",  fontsize=11)
    ax.set_title(title, fontsize=11, pad=6)
    return im


def plot_slice(ax, JSI, signal_bin, title,
               color_theory='#E06C00', color_data='#4C9BE8'):
    """
    Marginal bar chart: coincidences vs idler bin for one fixed signal bin.

    Orange stems = clean theory.
    Blue dots    = physically sampled data with sqrt(N) error bars.

    Noise model (textbook Sec. 6.2.2):
      True coincidences : Poisson(mean * eta_A * eta_B)   — beam-splitter loss
      Accidentals       : NegativeBinomial(M_modes, p_nb) — thermal SPDC stats
    """
    ji     = signal_bin - 1
    theory = JSI[ji, :]               # expected counts for this signal bin

    # (a) True coincidences — Poisson, attenuated by detector efficiencies
    signal_mean   = np.maximum(theory - c0_inside, 0.0) * eta_A * eta_B
    signal_counts = rng.poisson(signal_mean)

    # (b) Accidental coincidences — negative binomial (thermal / super-Poissonian)
    #     NegBin(n, p): mean = n*(1-p)/p  =>  p = M_modes / (M_modes + c0_inside)
    p_nb              = M_modes / (M_modes + c0_inside)
    accidental_counts = rng.negative_binomial(M_modes, p_nb, size=theory.shape)

    sampled = (signal_counts + accidental_counts).astype(float)
    errors  = np.sqrt(sampled)
    errors[errors == 0] = 1.0   # floor error bar at 1 for zero-count bins

    x = np.arange(1, N_bins + 1)

    # Orange theory stems
    _, stemlines, baseline = ax.stem(
        x, theory, linefmt=color_theory, markerfmt=' ', basefmt='k-'
    )
    plt.setp(stemlines, linewidth=3.5, alpha=0.85)
    plt.setp(baseline,  linewidth=0.8)

    # Blue data dots with error bars
    ax.errorbar(
        x, sampled, yerr=errors,
        fmt='o', color=color_data,
        markersize=5, linewidth=1.2, capsize=3,
        zorder=5,
    )

    ax.set_xticks(x);  ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Idler Bin",          fontsize=11)
    ax.set_ylabel("Coincidences (2 s)", fontsize=10)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlim(0.3, N_bins + 0.7)


# ============================================================
# Figure 1 — 2x2 JSI grid: 5a, 5b, 5c, drifting
# ============================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

jsi_panels = [
    (JSI_a, r"(a) Unmodulated  ($\Delta_A = \Delta_B = 0$)"),
    (JSI_b, r"(b) In-phase  ($\Delta = 1.42$ rad, $\phi_A = \phi_B$)"),
    (JSI_c, r"(c) Out-of-phase  ($\Delta = 1.42$ rad, $\phi_A - \phi_B = \pi$)"),
    (JSI_d, r"(d) Drifting / failed sync  (incoherent average)"),
]

for ax, (JSI, title) in zip(axes1.ravel(), jsi_panels):
    vmax = vmax_shared if shared_scale else None
    im1  = plot_JSI(ax, JSI, title, vmax=vmax)

cb1 = fig1.colorbar(im1, ax=axes1.ravel().tolist(), shrink=0.6, pad=0.02)
cb1.set_label("Coincidences (2 s integration)", fontsize=11)
fig1.suptitle(
    "Simulated JSI — Nonlocal Modulation Cancelation\n"
    r"($\Omega/2\pi = 19$ GHz,  $\Gamma = 12$ GHz,  $\Delta = 1.42$ rad)"
    + ("  [shared colour scale]" if shared_scale else "  [independent colour scales]"),
    fontsize=12,
)
fig1.savefig("fig5_jsi_all.png", dpi=150, bbox_inches="tight")
print("Saved: fig5_jsi_all.png")
plt.show()

# ============================================================
# Figure 2 — 3x2 slice grid: JSI + marginal chart for 5a, 5b, 5c
# ============================================================
fig2, axes2 = plt.subplots(3, 2, figsize=(12, 13), constrained_layout=True)

slice_rows = [
    (JSI_a, "(a) Unmodulated",   r"(a) Slice — unmodulated",
     np.max(JSI_a)),
    (JSI_b, r"(b) In-phase  ($\phi_A = \phi_B$)",
     r"(b) Slice — in-phase",    np.max(JSI_b)),
    (JSI_c, r"(c) Out-of-phase  ($\phi_A - \phi_B = \pi$)",
     r"(c) Slice — out-of-phase", np.max(JSI_c)),
]

for row, (JSI, jsi_title, slice_title, vmax_row) in enumerate(slice_rows):
    # Left: JSI heatmap with white line marking the sliced column
    im2 = plot_JSI(axes2[row, 0], JSI, jsi_title, vmax=vmax_row)
    axes2[row, 0].axvline(
        x=slice_signal_bin - 1, color='white', linewidth=2, alpha=0.7
    )
    fig2.colorbar(im2, ax=axes2[row, 0], fraction=0.046, pad=0.04)

    # Right: marginal slice bar chart
    plot_slice(
        axes2[row, 1], JSI, slice_signal_bin,
        slice_title + "  (signal bin {:d})".format(slice_signal_bin),
    )

fig2.suptitle(
    "JSI and Marginal Slices at Signal Bin {:d}\n"
    r"($\Omega/2\pi = 19$ GHz,  $\Gamma = 12$ GHz,  $\Delta = 1.42$ rad)".format(
        slice_signal_bin),
    fontsize=12,
)
fig2.savefig("fig5_slices.png", dpi=150, bbox_inches="tight")
print("Saved: fig5_slices.png")
plt.show()

# ============================================================
# Figure 3 — Phase sweep: R(5,5) vs phi_diff from 0 to 2*pi
# Shows the full modulation cycle from in-phase to out-of-phase
# ============================================================
print("Computing phase sweep (200 points)...")
R_norm_sweep = compute_Rjk(center_bin, center_bin, 0.0, 0.0, 0.0)
phi_range    = np.linspace(0, 2 * np.pi, 200)
R_sweep      = np.array([
    compute_Rjk(center_bin, center_bin, Delta, Delta, phi) / R_norm_sweep
    for phi in phi_range
])

# Annotate the three experimental operating points
phi_inphase    = 0.0
phi_outofphase = np.pi
R_inphase      = compute_Rjk(center_bin, center_bin, Delta, Delta, phi_inphase)    / R_norm_sweep
R_outofphase   = compute_Rjk(center_bin, center_bin, Delta, Delta, phi_outofphase) / R_norm_sweep
R_unmodulated  = 1.0   # by definition (normalised)

fig3, ax3 = plt.subplots(figsize=(8, 4.5))
ax3.plot(phi_range / np.pi, R_sweep, color='steelblue', linewidth=2,
         label=r'$R_{55}(\phi_\mathrm{diff})$')

# Mark key operating points
ax3.scatter([phi_inphase / np.pi],    [R_inphase],
            color='darkorange', s=80, zorder=5,
            label=r'In-phase ($\phi=0$):  $R={:.2f}$'.format(R_inphase))
ax3.scatter([phi_outofphase / np.pi], [R_outofphase],
            color='green', s=80, zorder=5,
            label=r'Out-of-phase ($\phi=\pi$):  $R={:.4f}$'.format(R_outofphase))
ax3.axhline(y=R_unmodulated, color='grey', linestyle='--', linewidth=1.2,
            label='Unmodulated baseline ($R=1$)')

ax3.set_xlabel(r"Phase difference  $(\phi_A - \phi_B) \ / \ \pi$", fontsize=12)
ax3.set_ylabel(r"Normalised coincidence rate  $R_{55}$",            fontsize=12)
ax3.set_title(
    r"Phase sweep at centre bin (5,5) — $\Delta = 1.42$ rad, $\Omega/2\pi = 19$ GHz",
    fontsize=12,
)
ax3.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax3.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig("fig5_phase_sweep.png", dpi=150, bbox_inches="tight")
print("Saved: fig5_phase_sweep.png")
plt.show()
