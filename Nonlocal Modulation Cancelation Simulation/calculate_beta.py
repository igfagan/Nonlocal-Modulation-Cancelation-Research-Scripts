"""
calculate_beta.py

Offline post-processing script for OSA trace bundles produced by the OSA
trace-pull script (all_traces.npz + metadata.json in OUTPUT_DIR).

Extracts the phase-modulation index (beta) from an optical spectrum that
shows a carrier plus RF-spaced sidebands, by fitting the measured peak
powers to the Bessel-function model of a sinusoidally phase-modulated
carrier:

    P_n = A * J_n(beta)^2

where P_n is the optical power in the n-th sideband (n=0 is the carrier),
J_n is the n-th order Bessel function of the first kind, A is an overall
power scale (~ input carrier power), and beta is the modulation index.

Run this against the OUTPUT_DIR produced by the OSA pull script.
"""

import json
import pathlib

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import curve_fit


# =========================
# User settings
# =========================
OUTPUT_DIR = pathlib.Path("PPCL_EOPM_RF-10ghz_-15-10dbm_7dBmlaser_200GHz_span")
NPZ_NAME = "all_traces.npz"

RF_FREQ_HZ = 10.0e9          # modulation frequency used to locate sidebands
N_ORDERS_MAX = 4             # search orders n = -N_ORDERS_MAX .. +N_ORDERS_MAX
SIDEBAND_SEARCH_FRAC = 0.25  # search window = +/- this fraction of RF_FREQ_HZ around each expected peak

CARRIER_WAVELENGTH_NM = None  # None = auto-detect (global max of trace). Set a value (e.g. 1550.12)
                              # if beta is large enough that a sideband can exceed the carrier.
CARRIER_SEARCH_WINDOW_HZ = RF_FREQ_HZ * 0.5  # window used when refining carrier position

MIN_SNR_DB = 3.0             # minimum peak height above noise floor to include an order in the fit

MAKE_BETA_GIF = False        # animate the per-frame beta_fit PNGs into a GIF
BETA_GIF_NAME = "beta_fit_evolution.gif"
GIF_FPS = 5

C_LIGHT = 299792458.0        # m/s

BETA_FITS_SUBDIR = "beta_fits"
RESULTS_JSON_NAME = "beta_results.json"
TREND_PLOT_NAME = "beta_vs_frame.png"


def load_trace_bundle(npz_path: pathlib.Path):
    data = np.load(npz_path, allow_pickle=True)
    x_m = data["x_m"]
    y = data["y"]
    timestamps = data["timestamps"]

    if x_m.dtype == object:
        traces = [(np.asarray(x_m[i], dtype=float), np.asarray(y[i], dtype=float)) for i in range(len(x_m))]
    else:
        traces = [(x_m[i], y[i]) for i in range(x_m.shape[0])]

    return traces, timestamps


def find_peak_near(freq_hz: np.ndarray, y_lin: np.ndarray, target_freq_hz: float, window_hz: float):
    mask = np.abs(freq_hz - target_freq_hz) <= (window_hz / 2.0)

    if not np.any(mask):
        return None

    idx_local = np.argmax(y_lin[mask])
    idx_global = np.flatnonzero(mask)[idx_local]

    return idx_global


def bessel_model(n, amplitude, beta):
    return amplitude * jv(n, beta) ** 2


def extract_beta_for_frame(x_m: np.ndarray, y_db: np.ndarray):
    freq_hz = C_LIGHT / x_m
    y_lin = 10.0 ** (y_db / 10.0)

    noise_floor_db = np.percentile(y_db, 10)
    threshold_db = noise_floor_db + MIN_SNR_DB

    if CARRIER_WAVELENGTH_NM is None:
        carrier_idx = int(np.argmax(y_db))
    else:
        approx_freq = C_LIGHT / (CARRIER_WAVELENGTH_NM * 1e-9)
        carrier_idx = find_peak_near(freq_hz, y_lin, approx_freq, CARRIER_SEARCH_WINDOW_HZ)
        if carrier_idx is None:
            carrier_idx = int(np.argmax(y_db))

    carrier_freq_hz = freq_hz[carrier_idx]
    search_window_hz = RF_FREQ_HZ * SIDEBAND_SEARCH_FRAC

    orders = []
    peak_indices = []
    peak_wavelengths_nm = []
    peak_power_db = []
    peak_power_lin = []

    for n in range(-N_ORDERS_MAX, N_ORDERS_MAX + 1):
        target_freq = carrier_freq_hz + n * RF_FREQ_HZ
        idx = find_peak_near(freq_hz, y_lin, target_freq, search_window_hz)

        if idx is None:
            continue

        if y_db[idx] < threshold_db:
            continue

        orders.append(n)
        peak_indices.append(int(idx))
        peak_wavelengths_nm.append(float(x_m[idx] * 1e9))
        peak_power_db.append(float(y_db[idx]))
        peak_power_lin.append(float(y_lin[idx]))

    orders = np.array(orders, dtype=float)
    peak_power_lin = np.array(peak_power_lin, dtype=float)

    if orders.size < 3:
        raise RuntimeError(
            f"Only {orders.size} peak(s) found above threshold ({threshold_db:.1f} dB); "
            "need at least 3 orders (carrier + sidebands) to fit beta."
        )

    beta0 = 1.0
    j0_guess = jv(0, beta0) ** 2
    a0 = peak_power_lin[np.argmin(np.abs(orders))] / j0_guess if j0_guess > 1e-6 else peak_power_lin.max()

    popt, pcov = curve_fit(
        bessel_model,
        orders,
        peak_power_lin,
        p0=[a0, beta0],
        bounds=([0, 0], [np.inf, 10]),
        maxfev=10000,
    )
    amplitude_fit, beta_fit = popt
    perr = np.sqrt(np.diag(pcov))
    beta_err = float(perr[1])

    model_power_lin = bessel_model(orders, amplitude_fit, beta_fit)

    result = {
        "beta": float(beta_fit),
        "beta_err": beta_err,
        "amplitude_fit": float(amplitude_fit),
        "carrier_wavelength_nm": float(x_m[carrier_idx] * 1e9),
        "orders_used": orders.astype(int).tolist(),
        "measured_power_db": peak_power_db,
        "model_power_db": (10.0 * np.log10(model_power_lin)).tolist(),
        "peak_wavelengths_nm": peak_wavelengths_nm,
    }

    return result, carrier_idx


def plot_frame_fit(x_m, y_db, result, out_png: pathlib.Path, title: str):
    fig, (ax_trace, ax_fit) = plt.subplots(1, 2, figsize=(14, 5))

    ax_trace.plot(x_m * 1e9, y_db, linewidth=0.8)
    ax_trace.scatter(result["peak_wavelengths_nm"], result["measured_power_db"], color="red", zorder=5, s=25)
    for n, wl, p_db in zip(result["orders_used"], result["peak_wavelengths_nm"], result["measured_power_db"]):
        ax_trace.annotate(f"n={n}", (wl, p_db), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax_trace.set_xlabel("Wavelength (nm)")
    ax_trace.set_ylabel("Level (dB)")
    ax_trace.set_title(title)
    ax_trace.grid(True, alpha=0.3)

    ax_fit.plot(result["orders_used"], result["measured_power_db"], "o", label="Measured")
    ax_fit.plot(result["orders_used"], result["model_power_db"], "-", label=f"Bessel fit (beta={result['beta']:.3f})")
    ax_fit.set_xlabel("Sideband order n")
    ax_fit.set_ylabel("Level (dB)")
    ax_fit.set_title("Measured vs. Bessel model")
    ax_fit.legend()
    ax_fit.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def build_gif(frame_pngs, out_gif: pathlib.Path, fps: int):
    images = [imageio.imread(f) for f in frame_pngs]
    imageio.mimsave(out_gif, images, fps=fps)


def main():
    npz_path = OUTPUT_DIR / NPZ_NAME
    traces, timestamps = load_trace_bundle(npz_path)

    fits_dir = OUTPUT_DIR / BETA_FITS_SUBDIR
    fits_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    frame_pngs = []

    for i, (x_m, y_db) in enumerate(traces):
        print(f"\nFrame {i}: fitting beta...")

        try:
            result, _ = extract_beta_for_frame(x_m, y_db)
        except RuntimeError as e:
            print(f"  Skipped: {e}")
            continue

        result["frame_index"] = i
        result["timestamp_unix"] = float(timestamps[i])
        all_results.append(result)

        print(f"  beta = {result['beta']:.4f} +/- {result['beta_err']:.4f}  "
              f"(orders used: {result['orders_used']})")

        out_png = fits_dir / f"frame_{i:04d}_beta_fit.png"
        plot_frame_fit(x_m, y_db, result, out_png, title=f"Frame {i:04d}")
        frame_pngs.append(out_png)

    with open(OUTPUT_DIR / RESULTS_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / RESULTS_JSON_NAME}")

    if MAKE_BETA_GIF and len(frame_pngs) > 1:
        gif_path = OUTPUT_DIR / BETA_GIF_NAME
        print(f"\nBuilding GIF from {len(frame_pngs)} frames...")
        build_gif(frame_pngs, gif_path, fps=GIF_FPS)
        print(f"GIF saved to {gif_path}")

    if len(all_results) > 1:
        frame_idx = [r["frame_index"] for r in all_results]
        betas = [r["beta"] for r in all_results]
        errs = [r["beta_err"] for r in all_results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(frame_idx, betas, yerr=errs, marker="o", linestyle="-")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Beta (modulation index)")
        ax.set_title("Beta vs. frame")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / TREND_PLOT_NAME, dpi=120)
        plt.close(fig)

        print(f"Trend plot saved to {OUTPUT_DIR / TREND_PLOT_NAME}")

    print("\nDone.")


if __name__ == "__main__":
    main()
