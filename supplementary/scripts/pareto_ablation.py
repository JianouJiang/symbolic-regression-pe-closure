#!/usr/bin/env python3
"""
Pareto front ablation study.

Evaluates multiple functional forms of increasing complexity against
the same validation datasets to justify the choice of the discovered expression.

All models including the original PySR expression are evaluated using the manuscript's ratio-of-tanh form
(complexity 21, 4 params). PySR originally discovered an exp-ln²-tanh variant that
differs outside the evaluation domain; the ratio-of-tanh form is adopted because it
recovers the correct asymptotic limits.

Functional forms tested:
1. Constant (C=1): P/ε = a
2. Linear (C=5): P/ε = a + b/y+
3. Log-linear (C=7): P/ε = a + b·ln(y+) + c/y+
4. Log-polynomial (C=12): P/ε = Σ ak·(ln y+)^k / y+
5. Rational (C=15): P/ε = (a + b·y+) / (c + y+)
6. STLSQ (C=14): Linear model of 14 basis functions
7. Original PySR (C=21): The recommended printed formula
"""
import json
import numpy as np
import os
from pathlib import Path
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_FILE = RESULTS_DIR / "pareto_ablation.json"


def pe_formula_c22(y_plus):
    """Printed original PySR ratio-of-tanh form (complexity 21, 4 params)."""
    yp = np.maximum(y_plus, 1e-10)
    alpha1 = 0.128
    alpha2 = 0.053
    alpha3 = 0.620
    alpha4 = 5.78
    return np.tanh(alpha1 * yp) / (np.tanh(alpha2 * yp - alpha3) + alpha4 / yp)


def model_constant(yp, a):
    return np.full_like(yp, a)

def model_linear(yp, a, b):
    return a + b / yp

def model_log_linear(yp, a, b, c):
    return a + b * np.log(yp) + c / yp

def model_log_polynomial(yp, a0, a1, a2, a3, a4):
    ln = np.log(yp)
    return (a0 + a1 * ln + a2 * ln**2 + a3 * ln**3 + a4 * ln**4) / yp + 1.0

def model_rational(yp, a, b, c, d):
    return (a + b * yp) / (c + yp) + d * np.exp(-yp / 10.0)

def model_tanh_exp(yp, a, b, c, d, e):
    return a * np.exp(-b / yp) * np.tanh(c / yp) + d + e / yp


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('-inf')


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_channel_training():
    all_yp, all_pe = [], []
    for re in [180, 550, 1000, 2000, 5200]:
        f = RESULTS_DIR / f"channel_Re{re}.npz"
        d = np.load(f)
        yp = d["y_plus"]
        pe = d["P_over_eps"]
        mask = (yp > 1) & (yp <= 150) & (pe > 0.01)
        all_yp.append(yp[mask])
        all_pe.append(pe[mask])
    return np.concatenate(all_yp), np.concatenate(all_pe)


def load_tbl_validation():
    bud_dir = RESULTS_DIR / "kth_tbl_budgets"
    datasets = {}
    for re_theta in [670, 1000, 1410, 2000, 2540, 3030, 3270, 3630, 3970, 4060]:
        fname = bud_dir / f"bud_{re_theta:04d}_dns_k.prof"
        if not fname.exists():
            continue
        rows = []
        with open(fname) as f:
            in_data = False
            for line in f:
                if line.strip().startswith("y/\\delta"):
                    in_data = True
                    continue
                if in_data:
                    vals = line.strip().split()
                    if len(vals) >= 7:
                        try:
                            rows.append([float(v) for v in vals[:9]])
                        except ValueError:
                            continue
        if not rows:
            continue
        data = np.array(rows)
        yp = data[:, 1]
        prod, diss = data[:, 3], data[:, 4]
        eps = -diss
        pe_raw = prod / np.maximum(eps, 1e-30)
        mask = (yp > 1) & (yp <= 150) & (eps > 1e-6) & (pe_raw > 0.01)
        datasets[re_theta] = (yp[mask], pe_raw[mask])
    return datasets


def load_pipe_validation():
    pipe_dir = RESULTS_DIR / "el_khoury_pipe"
    datasets = {}
    for re_tau in [180, 360, 550, 1000]:
        budgets = {}
        for comp in ["RR", "TT", "ZZ"]:
            fname = pipe_dir / f"{re_tau}_{comp}_Budget.dat"
            if not fname.exists():
                break
            rows = []
            with open(fname) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    try:
                        vals = [float(x) for x in line.split()]
                        if len(vals) >= 8:
                            rows.append(vals)
                    except ValueError:
                        continue
            budgets[comp] = np.array(rows)
        if len(budgets) < 3:
            continue
        yp = budgets["RR"][:, 1]
        P_k = 0.5 * (budgets["RR"][:, 2] + budgets["TT"][:, 2] + budgets["ZZ"][:, 2])
        eps_k = -0.5 * (budgets["RR"][:, 7] + budgets["TT"][:, 7] + budgets["ZZ"][:, 7])
        pe_raw = P_k / np.maximum(eps_k, 1e-30)
        mask = (yp > 1) & (yp <= 150) & (eps_k > 1e-6) & (pe_raw > 0.01)
        datasets[re_tau] = (yp[mask], pe_raw[mask])
    return datasets


def fit_and_evaluate(model_func, yp_train, pe_train, p0, name, complexity,
                     tbl_data, pipe_data, bounds=(-np.inf, np.inf)):
    """Fit model on training data and evaluate on all validation."""
    try:
        popt, _ = curve_fit(model_func, yp_train, pe_train, p0=p0,
                           maxfev=10000, bounds=bounds)
        pred_train = model_func(yp_train, *popt)
        r2_train = compute_r2(pe_train, pred_train)
        rmse_train = compute_rmse(pe_train, pred_train)
    except Exception as e:
        print(f"  {name}: fit failed: {e}")
        return None

    # TBL validation
    tbl_r2s = []
    for re_theta, (yp_v, pe_v) in tbl_data.items():
        try:
            pred_v = model_func(yp_v, *popt)
            r2_v = compute_r2(pe_v, pred_v)
            tbl_r2s.append(r2_v)
        except:
            pass

    # Pipe validation
    pipe_r2s = []
    for re_tau, (yp_v, pe_v) in pipe_data.items():
        try:
            pred_v = model_func(yp_v, *popt)
            r2_v = compute_r2(pe_v, pred_v)
            pipe_r2s.append(r2_v)
        except:
            pass

    result = {
        'name': name,
        'complexity': complexity,
        'n_params': len(popt),
        'params': [float(p) for p in popt],
        'R2_train': float(r2_train),
        'RMSE_train': rmse_train,
        'R2_tbl_mean': float(np.mean(tbl_r2s)) if tbl_r2s else None,
        'R2_tbl_std': float(np.std(tbl_r2s)) if tbl_r2s else None,
        'R2_tbl_n': len(tbl_r2s),
        'R2_pipe_mean': float(np.mean(pipe_r2s)) if pipe_r2s else None,
        'R2_pipe_n': len(pipe_r2s),
    }

    print(f"  {name:25s} C={complexity:2d} | Train R²={r2_train:.4f} | "
          f"TBL R²={np.mean(tbl_r2s):.4f}" if tbl_r2s else
          f"  {name:25s} C={complexity:2d} | Train R²={r2_train:.4f} | TBL: N/A",
          f" | Pipe R²={np.mean(pipe_r2s):.4f}" if pipe_r2s else "")

    return result


def main():
    print("=" * 70)
    print("PARETO FRONT ABLATION: Model complexity vs accuracy")
    print("=" * 70)

    yp, pe = load_channel_training()
    tbl_data = load_tbl_validation()
    pipe_data = load_pipe_validation()
    print(f"Training: {len(yp)} points")
    print(f"TBL validation: {len(tbl_data)} datasets")
    print(f"Pipe validation: {len(pipe_data)} datasets")

    results = []

    # 1. Constant
    r = fit_and_evaluate(model_constant, yp, pe, [1.0], "Constant", 1,
                        tbl_data, pipe_data)
    if r: results.append(r)

    # 2. Linear in 1/y+
    r = fit_and_evaluate(model_linear, yp, pe, [1.0, -10.0], "Linear (1/y+)", 5,
                        tbl_data, pipe_data)
    if r: results.append(r)

    # 3. Log-linear
    r = fit_and_evaluate(model_log_linear, yp, pe, [1.0, 0.1, -5.0],
                        "Log-linear", 7, tbl_data, pipe_data)
    if r: results.append(r)

    # 4. Log-polynomial
    r = fit_and_evaluate(model_log_polynomial, yp, pe, [1.0, 0.1, 0.01, 0.001, 0.0001],
                        "Log-polynomial", 12, tbl_data, pipe_data)
    if r: results.append(r)

    # 5. Rational
    r = fit_and_evaluate(model_rational, yp, pe, [1.0, 0.01, 5.0, 1.0],
                        "Rational + exp", 15, tbl_data, pipe_data)
    if r: results.append(r)

    # 6. tanh-exp
    r = fit_and_evaluate(model_tanh_exp, yp, pe, [2.0, 5.0, 25.0, 0.5, -1.0],
                        "tanh-exp (5 param)", 18, tbl_data, pipe_data)
    if r: results.append(r)

    # 7. Printed original PySR expression — no fitting needed, just evaluate
    pred_train_c22 = pe_formula_c22(yp)
    r2_train_c22 = compute_r2(pe, pred_train_c22)
    tbl_r2s_c22 = []
    for re_theta, (yp_v, pe_v) in tbl_data.items():
        pred_v = pe_formula_c22(yp_v)
        tbl_r2s_c22.append(compute_r2(pe_v, pred_v))
    pipe_r2s_c22 = []
    for re_tau, (yp_v, pe_v) in pipe_data.items():
        pred_v = pe_formula_c22(yp_v)
        pipe_r2s_c22.append(compute_r2(pe_v, pred_v))

    results.append({
        'name': 'Original PySR ratio-of-tanh',
        'complexity': 21,
        'n_params': 4,
        'params': [0.128, 0.053, 0.620, 5.78],
        'note': 'PySR originally discovered an exp-ln\u00b2-tanh form (complexity 22, 5 params) which was recast into this ratio-of-tanh form with correct asymptotic limits. The two forms differ outside the evaluation domain.',
        'R2_train': float(r2_train_c22),
        'RMSE_train': compute_rmse(pe, pred_train_c22),
        'R2_tbl_mean': float(np.mean(tbl_r2s_c22)),
        'R2_tbl_std': float(np.std(tbl_r2s_c22)),
        'R2_tbl_n': len(tbl_r2s_c22),
        'R2_pipe_mean': float(np.mean(pipe_r2s_c22)),
        'R2_pipe_n': len(pipe_r2s_c22),
    })
    print(f"  {'Original PySR':25s} C=21 | Train R²={r2_train_c22:.4f} | "
          f"TBL R²={np.mean(tbl_r2s_c22):.4f} | Pipe R²={np.mean(pipe_r2s_c22):.4f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':25s} {'C':>3s} {'#P':>3s} {'Train R²':>9s} {'TBL R²':>9s} {'Pipe R²':>9s}")
    print("-" * 70)
    for r in results:
        tbl = f"{r['R2_tbl_mean']:.4f}" if r['R2_tbl_mean'] else "N/A"
        pipe = f"{r['R2_pipe_mean']:.4f}" if r['R2_pipe_mean'] else "N/A"
        print(f"{r['name']:25s} {r['complexity']:3d} {r['n_params']:3d} "
              f"{r['R2_train']:9.4f} {tbl:>9s} {pipe:>9s}")

    summary = {
        'description': 'Pareto front ablation: complexity vs accuracy on same train/test',
        'n_train': len(yp),
        'n_tbl_datasets': len(tbl_data),
        'n_pipe_datasets': len(pipe_data),
        'models': results,
    }

    with open(OUT_FILE, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
