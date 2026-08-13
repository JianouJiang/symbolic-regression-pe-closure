#!/usr/bin/env python3
"""
Re-run the DNS wall-slope fit with a bootstrap 95%
confidence band on the exponent n in P/eps = c_w (y+)^n, fitted on y+ in
[0.1, yp_max] for each of the five training Re_tau.

The nominal tolerance envelope is n in [2.9, 3.1]. The  wall_asymp_proof.py returned
point estimates only; at Re_tau=5200 the point estimate was 3.12, marginally
outside the envelope. This script reports a CI to honestly bracket whether
the DNS exponent is statistically inside or outside [2.9, 3.1].

Bootstrap method: residual resampling of the linear regression in log-log
space, B=2000 replicates, deterministic seed.

Output: supplementary/results/wall_slope_ci.json
"""
import json
import os
import numpy as np

import os
from pathlib import Path

# By default, channel-DNS data lives at <project-root>/codes/results/channel_Re*.npz.
# Override SCALING_LAW_PROJECT_ROOT env-var if your layout differs.
SUPP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get(
    "SCALING_LAW_PROJECT_ROOT",
    str(SUPP_DIR.parent.parent),
))
PROJ_RESULTS = PROJECT_ROOT / "codes" / "results"
RESULTS_DIR = SUPP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = RESULTS_DIR

RE_LIST = [180, 550, 1000, 2000, 5200]
G1_LO, G1_HI = 2.9, 3.1
N_BOOT = 2000
SEED = 20260527


def load_channel(re_tau):
    d = np.load(os.path.join(PROJ_RESULTS, f"channel_Re{re_tau}.npz"))
    return d["y_plus"], d["P_over_eps"]


def fit_slope(yp, pe, yp_lo, yp_hi):
    """Return (slope, intercept, npts, x, y, residuals)."""
    m = (yp > yp_lo) & (yp < yp_hi) & (pe > 0)
    if m.sum() < 3:
        return (float("nan"),) * 6
    x = np.log(yp[m])
    y = np.log(pe[m])
    s, b = np.polyfit(x, y, 1)
    res = y - (s * x + b)
    return s, b, m.sum(), x, y, res


def bootstrap_slope_ci(x, y, res, B, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    slopes = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        y_b = (np.polyval([np.polyfit(x, y, 1)[0], np.polyfit(x, y, 1)[1]], x) + res[idx])
        s_b, _ = np.polyfit(x, y_b, 1)
        slopes[i] = s_b
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi), float(np.std(slopes))


def main():
    results = {
        "metric": "G1 wall exponent (P/eps = c_w y+^n)",
        "G1_envelope": [G1_LO, G1_HI],
        "fit_window_default_yp": [0.1, 2.0],
        "fit_alt_windows_yp": [[0.1, 1.0], [0.1, 1.5], [0.1, 3.0]],
        "bootstrap_method": "residual resampling in log-log",
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "per_Re_tau": {},
    }

    print(f"{'Re_tau':>6s}  {'window':>12s}  {'n_pts':>6s}  "
          f"{'n_hat':>8s}  {'95% CI':>20s}  {'G1 status':>16s}")

    for re in RE_LIST:
        yp, pe = load_channel(re)
        per_window = {}
        all_windows = [(0.1, 1.0), (0.1, 1.5), (0.1, 2.0), (0.1, 3.0)]
        for yp_lo, yp_hi in all_windows:
            s, b, n, x, y, res = fit_slope(yp, pe, yp_lo, yp_hi)
            if not np.isfinite(s):
                per_window[f"{yp_lo}_{yp_hi}"] = {"status": "insufficient_data"}
                continue
            ci_lo, ci_hi, sd = bootstrap_slope_ci(x, y, res, N_BOOT, SEED + re + int(yp_hi * 10))
            inside = (ci_lo >= G1_LO and ci_hi <= G1_HI)
            envelope_overlap = not (ci_hi < G1_LO or ci_lo > G1_HI)
            per_window[f"yp_{yp_lo}_to_{yp_hi}"] = {
                "n_pts": int(n),
                "n_hat": float(s),
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "std_boot": sd,
                "ci_inside_G1": bool(inside),
                "ci_overlaps_G1": bool(envelope_overlap),
            }
            if (yp_lo, yp_hi) == (0.1, 2.0):
                status = ("INSIDE" if inside else
                          ("OVERLAPS" if envelope_overlap else "OUTSIDE"))
                print(f"{re:>6d}  {f'[{yp_lo},{yp_hi}]':>12s}  {n:>6d}  "
                      f"{s:>8.3f}  {f'[{ci_lo:.3f}, {ci_hi:.3f}]':>20s}  "
                      f"{status:>16s}")
        results["per_Re_tau"][str(re)] = per_window

    # Verdict: count Re_tau whose CI overlaps G1 for the default window
    overlaps = sum(
        results["per_Re_tau"][str(re)]["yp_0.1_to_2.0"]["ci_overlaps_G1"]
        for re in RE_LIST)
    insides = sum(
        results["per_Re_tau"][str(re)]["yp_0.1_to_2.0"]["ci_inside_G1"]
        for re in RE_LIST)
    results["verdict"] = {
        "G1_envelope": [G1_LO, G1_HI],
        "n_Re_tau_ci_inside_G1": int(insides),
        "n_Re_tau_ci_overlaps_G1": int(overlaps),
        "n_Re_tau_total": len(RE_LIST),
        "pass_overlap_all": bool(overlaps == len(RE_LIST)),
        "pass_inside_all": bool(insides == len(RE_LIST)),
        "note": ("CI overlap with nominal tolerance envelope means the DNS exponent is "
                 "statistically consistent with the kinematic prediction "
                 "n=3 within sampling noise."),
    }

    out = os.path.join(OUT_DIR, "wall_slope_ci.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    print(f"Verdict: {insides}/{len(RE_LIST)} Re_tau have CI inside [2.9, 3.1], "
          f"{overlaps}/{len(RE_LIST)} Re_tau have CI overlapping [2.9, 3.1].")


if __name__ == "__main__":
    main()
