#!/usr/bin/env python3
"""
Wall-region-weighted refit of the derived closure
Eq. (M16): F(y+) = tanh^2(a1 y+) / [tanh(a2 y+ - a3) + a4/y+]

Goal: drive the predicted wall coefficient c_w = a1^2 / a4 to within ~20%
of the DNS-measured c_w (fit on y+ in [0.1, 2]) at all five Re_tau, by
re-fitting (a1, a2, a3, a4) with a wall-emphasising weight on the loss.

Reproducibility: deterministic seed; OMP_NUM_THREADS=2.

Output: supplementary/results/refit_wall_weighted.json
"""
import json
import os
import numpy as np
from scipy.optimize import minimize

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

# Baseline (track_b Candidate A) constants —  reference
ALPHA_BASELINE = [
    0.09385000189632862, 0.053325420623880855,
    0.6173156634120224, 5.054611667649887,
]


def F_new(yp, alpha):
    a1, a2, a3, a4 = alpha
    yp = np.maximum(yp, 1e-12)
    return np.tanh(a1 * yp)**2 / (np.tanh(a2 * yp - a3) + a4 / yp)


def load_channel(re_tau):
    f = os.path.join(PROJ_RESULTS, f"channel_Re{re_tau}.npz")
    d = np.load(f)
    yp = d["y_plus"]
    pe = d["P_over_eps"]
    return yp, pe


def measure_cw_dns(re_tau, yp_max=2.0):
    """DNS wall coefficient: fit P/eps ~ c_w (y+)^n on y+ in [0.1, yp_max]."""
    yp, pe = load_channel(re_tau)
    m = (yp > 0.1) & (yp < yp_max) & (pe > 0)
    if m.sum() < 3:
        return float("nan"), float("nan"), 0
    n_fit, log_c = np.polyfit(np.log(yp[m]), np.log(pe[m]), 1)
    return float(np.exp(log_c)), float(n_fit), int(m.sum())


def build_training_pool(yp_lo=1.0, yp_hi_cap=150):
    yps, pes, rets, srcs = [], [], [], []
    for re in RE_LIST:
        yp, pe = load_channel(re)
        m = (yp > yp_lo) & (yp <= min(yp_hi_cap, 0.3 * re)) & (pe > 0.01)
        yps.append(yp[m])
        pes.append(pe[m])
        rets.append(np.full(m.sum(), re))
        srcs.append(np.full(m.sum(), f"Re{re}"))
    return (np.concatenate(yps), np.concatenate(pes),
            np.concatenate(rets), np.concatenate(srcs))


def build_wall_pool(yp_lo=0.1, yp_hi=4.0):
    """Wall-region points used to anchor c_w; emphasised in the weighted loss."""
    yps, pes = [], []
    for re in RE_LIST:
        yp, pe = load_channel(re)
        m = (yp > yp_lo) & (yp < yp_hi) & (pe > 0)
        yps.append(yp[m])
        pes.append(pe[m])
    return np.concatenate(yps), np.concatenate(pes)


def fit_weighted(F, x0, bounds, yp_t, pe_t, yp_w, pe_w, wall_weight):
    n_t, n_w = len(yp_t), len(yp_w)

    def objective(alpha):
        try:
            pt = F(yp_t, alpha)
            pw = F(yp_w, alpha)
            if not (np.all(np.isfinite(pt)) and np.all(np.isfinite(pw))):
                return 1e10
            # Log-space residual for wall pool (puts wall coefficient on equal
            # footing across Re_tau and avoids vanishing contribution from
            # small y+ values).
            log_res_w = np.log(np.maximum(pw, 1e-30)) - np.log(np.maximum(pe_w, 1e-30))
            lin_res_t = pt - pe_t
            loss = (np.sum(lin_res_t**2) / n_t
                    + wall_weight * np.sum(log_res_w**2) / n_w)
            return loss
        except Exception:
            return 1e10

    r = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 30000, "ftol": 1e-15})
    return r


def metrics(alpha, yp, pe):
    pred = F_new(yp, alpha)
    ss_res = float(np.sum((pe - pred) ** 2))
    ss_tot = float(np.sum((pe - pe.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((pe - pred) ** 2)))
    return r2, rmse


def cw_pred(alpha):
    a1, _, _, a4 = alpha
    return a1 * a1 / a4


def run():
    yp_t, pe_t, _, _ = build_training_pool()
    yp_w, pe_w = build_wall_pool(yp_lo=0.1, yp_hi=4.0)

    bounds = [(0.01, 0.6), (0.01, 0.3), (0.1, 2.0), (1.0, 25.0)]
    sweeps = [0.0, 1.0, 5.0, 20.0, 100.0]

    results = {
        "F_form": "tanh^2(a1 y+)/[tanh(a2 y+ - a3) + a4/y+]",
        "channels": [f"Re_tau={r}" for r in RE_LIST],
        "wall_pool": {"yp_range_plus": [0.1, 4.0], "n_points": int(len(yp_w))},
        "training_pool": {"yp_range_plus": [1.0, "min(150, 0.3 Re_tau)"],
                          "n_points": int(len(yp_t))},
        "loss": "MSE(buffer/log) + wall_weight * MSE(log(F_pred)-log(P/eps), wall)",
        "baseline_alpha": ALPHA_BASELINE,
        "baseline_cw_pred": cw_pred(ALPHA_BASELINE),
        "sweeps": [],
    }

    # DNS reference c_w per Re_tau (independent of fit)
    dns_ref = {}
    for re in RE_LIST:
        cw, n, npts = measure_cw_dns(re)
        dns_ref[str(re)] = {"c_w_dns": cw, "n_dns_wall": n, "n_pts": npts}
    results["dns_reference_cw_yp_lt_2"] = dns_ref
    cw_dns_mean = float(np.mean([d["c_w_dns"] for d in dns_ref.values()]))
    results["dns_reference_cw_mean"] = cw_dns_mean

    for w in sweeps:
        fit = fit_weighted(F_new, list(ALPHA_BASELINE), bounds,
                           yp_t, pe_t, yp_w, pe_w, w)
        alpha_fit = [float(x) for x in fit.x]
        cw_pred_val = cw_pred(alpha_fit)
        # Compute global R^2 (training pool) and per-Re_tau R^2
        r2_train, rmse_train = metrics(alpha_fit, yp_t, pe_t)
        per_re_r2 = {}
        for re in RE_LIST:
            yp, pe = load_channel(re)
            m = (yp > 1.0) & (yp <= min(150, 0.3 * re)) & (pe > 0.01)
            r2, rmse = metrics(alpha_fit, yp[m], pe[m])
            per_re_r2[str(re)] = {"R2": r2, "RMSE": rmse, "n": int(m.sum())}
        wall_rel = 100.0 * abs(cw_pred_val - cw_dns_mean) / cw_dns_mean
        results["sweeps"].append({
            "wall_weight": w,
            "alpha_fit": alpha_fit,
            "converged": bool(fit.success),
            "loss_value": float(fit.fun),
            "cw_pred": cw_pred_val,
            "cw_rel_err_vs_dns_mean_percent": float(wall_rel),
            "R2_train_global": float(r2_train),
            "RMSE_train_global": float(rmse_train),
            "per_re_R2": per_re_r2,
        })

    # Pick the winner: smallest c_w error subject to global R^2 >= 0.95
    winner = None
    best_err = float("inf")
    for s in results["sweeps"]:
        if s["R2_train_global"] < 0.95:
            continue
        if s["cw_rel_err_vs_dns_mean_percent"] < best_err:
            best_err = s["cw_rel_err_vs_dns_mean_percent"]
            winner = s
    results["winner"] = winner
    results["cw_target_pass"] = bool(winner is not None and
                                       winner["cw_rel_err_vs_dns_mean_percent"] <= 20.0)
    results["cw_target_note"] = (
        "If no weight brings c_w within 20%, the closure predicts the wall "
        "EXPONENT (y+^3) exactly by construction, but the wall COEFFICIENT "
        "remains empirical (Mansour-Kim-Moin 1988 also empirical for the same "
        "coefficient). Manuscript will state this honestly."
    )

    out = os.path.join(OUT_DIR, "refit_wall_weighted.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")

    print(f"\nDNS reference c_w (fit on y+ in [0.1,2], mean over 5 Re_tau): "
          f"{cw_dns_mean:.4e}")
    print(f"Baseline () c_w_pred = a1^2/a4 = "
          f"{cw_pred(ALPHA_BASELINE):.4e}  "
          f"(rel.err = {100*abs(cw_pred(ALPHA_BASELINE)-cw_dns_mean)/cw_dns_mean:.1f}%)")
    print()
    print(f"{'wall_w':>8s}  {'c_w_pred':>12s}  {'rel.err [%]':>12s}  "
          f"{'R2_train':>10s}")
    for s in results["sweeps"]:
        print(f"{s['wall_weight']:>8.1f}  {s['cw_pred']:>12.4e}  "
              f"{s['cw_rel_err_vs_dns_mean_percent']:>12.2f}  "
              f"{s['R2_train_global']:>10.5f}")
    if winner:
        print(f"\nWinner (cw_err -> 0 with R2>=0.95):  wall_weight = "
              f"{winner['wall_weight']:.1f},  cw_rel.err = "
              f"{winner['cw_rel_err_vs_dns_mean_percent']:.2f}%")
        print(f"target (cw within 20% of DNS): "
              f"{'PASS' if results['cw_target_pass'] else 'FAIL'}")


if __name__ == "__main__":
    run()
