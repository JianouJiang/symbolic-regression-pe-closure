#!/usr/bin/env python3
"""Executable contracts for the wall series and the AEH extension.

The script deliberately refits alpha_4 with the *printed practical M16*
(alpha_1, alpha_2, alpha_3) triplet.  This prevents reuse of the earlier AEH
result, which was generated with a different coefficient triplet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


HERE = Path(__file__).resolve().parent
NODE = HERE
ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "audit_config.json"
DEFAULT_OUTPUT = NODE / "results" / "wall_aeh_contracts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def closure(y_plus: np.ndarray, power: int, alpha: np.ndarray) -> np.ndarray:
    a1, a2, a3, a4 = alpha
    y = np.asarray(y_plus, dtype=float)
    return np.tanh(a1 * y) ** power / (
        np.tanh(a2 * y - a3) + a4 / y
    )


def wall_series(alpha: list[float]) -> dict:
    """Return the analytic series coefficients through the first correction.

    With t=tanh(alpha_3),
      F_1 = a1/a4*y^2 + a1*t/a4^2*y^3 + O(y^4)
      F_2 = a1^2/a4*y^3 + a1^2*t/a4^2*y^4 + O(y^5).
    """
    a1, _a2, a3, a4 = alpha
    t = float(np.tanh(a3))
    return {
        "tanh_alpha3": t,
        "pysr_linear_tanh": {
            "leading_power": 2,
            "coefficient_y2": a1 / a4,
            "coefficient_y3": a1 * t / a4**2,
            "remainder_after_reported_terms": "O((y+)^4)",
        },
        "m16_squared_tanh": {
            "leading_power": 3,
            "coefficient_y3": a1**2 / a4,
            "coefficient_y4": a1**2 * t / a4**2,
            "remainder_after_reported_terms": "O((y+)^5)",
        },
    }


def load_channel(re_tau: int) -> tuple[np.ndarray, np.ndarray, Path]:
    path = ROOT / "codes" / "results" / f"channel_Re{re_tau}.npz"
    with np.load(path) as data:
        return (
            np.asarray(data["y_plus"], dtype=float),
            np.asarray(data["P_over_eps"], dtype=float),
            path,
        )


def fit_alpha4_by_profile(config: dict) -> tuple[list[dict], dict[str, str]]:
    practical = np.asarray(config["reported_formulas"]["m16"]["alpha"], dtype=float)
    a1, a2, a3, _ = practical
    low, high = config["aeh"]["alpha4_bounds"]
    mask_cfg = config["mask"]
    rows: list[dict] = []
    source_hashes: dict[str, str] = {}

    for re_tau in config["training_re_tau"]:
        y, target, source = load_channel(int(re_tau))
        source_hashes[str(source.relative_to(ROOT))] = sha256(source)
        y_max = min(
            mask_cfg["y_plus_max_absolute"],
            mask_cfg["y_plus_max_re_fraction"] * re_tau,
        )
        mask = (
            (y > mask_cfg["y_plus_min_exclusive"])
            & (y <= y_max)
            & (target > mask_cfg["p_over_epsilon_min"])
            & np.isfinite(y)
            & np.isfinite(target)
        )
        ym = y[mask]
        zm = target[mask]

        def objective(alpha4: float) -> float:
            pred = closure(ym, 2, np.array([a1, a2, a3, alpha4]))
            if not np.all(np.isfinite(pred)):
                return float("inf")
            return float(np.mean((zm - pred) ** 2))

        optimum = minimize_scalar(
            objective,
            bounds=(float(low), float(high)),
            method="bounded",
            options={"xatol": 1e-12},
        )
        alpha4 = float(optimum.x)
        pred = closure(ym, 2, np.array([a1, a2, a3, alpha4]))
        ss_res = float(np.sum((zm - pred) ** 2))
        ss_tot = float(np.sum((zm - np.mean(zm)) ** 2))
        rows.append(
            {
                "Re_tau": int(re_tau),
                "alpha4": alpha4,
                "R2": 1.0 - ss_res / ss_tot,
                "RMSE": float(np.sqrt(np.mean((zm - pred) ** 2))),
                "n": int(mask.sum()),
                "actual_y_plus_range": [float(ym.min()), float(ym.max())],
                "optimizer_success": bool(optimum.success),
            }
        )
    return rows, source_hashes


def regress_aeh(rows: list[dict], config: dict) -> dict:
    re_ref = float(config["aeh"]["re_tau_reference"])
    re_tau = np.array([row["Re_tau"] for row in rows], dtype=float)
    alpha4 = np.array([row["alpha4"] for row in rows], dtype=float)
    x = np.log(re_tau / re_ref)
    alpha5, alpha4_0 = np.polyfit(x, alpha4, 1)
    pred = alpha4_0 + alpha5 * x
    r2 = 1.0 - np.sum((alpha4 - pred) ** 2) / np.sum(
        (alpha4 - np.mean(alpha4)) ** 2
    )

    rng = np.random.default_rng(int(config["seed"]))
    target_replicates = int(config["aeh"]["bootstrap_replicates"])
    boot: list[np.ndarray] = []
    while len(boot) < target_replicates:
        index = rng.integers(0, len(rows), len(rows))
        if np.unique(x[index]).size < 2:
            continue
        coef = np.polyfit(x[index], alpha4[index], 1)
        if np.all(np.isfinite(coef)):
            boot.append(coef)
    boot_arr = np.asarray(boot)

    return {
        "formula": "alpha4(Re_tau)=alpha4_0+alpha5*ln(Re_tau/1000)",
        "Re_tau_reference": re_ref,
        "fitted_range": [float(re_tau.min()), float(re_tau.max())],
        "alpha4_0": float(alpha4_0),
        "alpha4_0_bootstrap_95CI": np.percentile(
            boot_arr[:, 1], [2.5, 97.5]
        ).tolist(),
        "alpha5": float(alpha5),
        "alpha5_bootstrap_95CI": np.percentile(
            boot_arr[:, 0], [2.5, 97.5]
        ).tolist(),
        "R2_five_profile_regression": float(r2),
        "bootstrap_unit": "Re_tau profile (paired Re_tau, fitted alpha4)",
        "bootstrap_replicates": target_replicates,
    }


def admissibility(alpha2: float, alpha3: float, aeh: dict, config: dict) -> dict:
    y_zero = alpha3 / alpha2
    optimum = minimize_scalar(
        lambda y: -y * np.tanh(alpha3 - alpha2 * y),
        bounds=(np.finfo(float).eps, y_zero),
        method="bounded",
    )
    y_critical = float(optimum.x)
    a_critical = float(-optimum.fun)
    alpha4_0 = float(aeh["alpha4_0"])
    alpha5 = float(aeh["alpha5"])
    re_ref = float(aeh["Re_tau_reference"])
    fitted_min, fitted_max = aeh["fitted_range"]

    def amplitude(re_tau: float) -> float:
        return alpha4_0 + alpha5 * np.log(re_tau / re_ref)

    if alpha5 > 0:
        global_re_lower = re_ref * np.exp((a_critical - alpha4_0) / alpha5)
        admissible_re_statement = "Re_tau > Re_tau_lower_global"
    elif alpha5 < 0:
        global_re_lower = None
        admissible_re_statement = (
            "No ultra-high-Re guarantee: A(Re_tau) eventually crosses the threshold"
        )
    else:
        global_re_lower = 0.0 if alpha4_0 > a_critical else None
        admissible_re_statement = "Re-independent admissibility"

    probes = [fitted_min, fitted_max] + list(
        config["aeh"]["extrapolation_probe_re_tau"]
    )
    return {
        "condition": (
            "A(Re_tau) > max_{0<y+<alpha3/alpha2} "
            "y+*tanh(alpha3-alpha2*y+)"
        ),
        "alpha3_over_alpha2": y_zero,
        "critical_y_plus": y_critical,
        "A_critical": a_critical,
        "Re_tau_lower_global": (
            float(global_re_lower) if global_re_lower is not None else None
        ),
        "admissible_re_statement": admissible_re_statement,
        "probes": [
            {
                "Re_tau": float(re_tau),
                "A": float(amplitude(float(re_tau))),
                "margin_A_minus_Acritical": float(
                    amplitude(float(re_tau)) - a_critical
                ),
                "evidence_status": (
                    "fitted/validated range"
                    if fitted_min <= re_tau <= fitted_max
                    else "algebraic extrapolation only"
                ),
            }
            for re_tau in probes
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    rows, source_hashes = fit_alpha4_by_profile(config)
    aeh = regress_aeh(rows, config)
    m16 = config["reported_formulas"]["m16"]
    pysr = config["reported_formulas"]["pysr"]
    a1, a2, a3, _a4 = m16["alpha"]
    domain = admissibility(float(a2), float(a3), aeh, config)

    out = {
        "schema_version": "1.0",
        "script_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(args.config),
        "source_sha256": source_hashes,
        "coefficient_level_kinematics": {
            "velocity_expansions": (
                "u'+=a1*y+ + a2*(y+)^2+...; v'+=b2*(y+)^2+b3*(y+)^3+...; "
                "w'+=c1*y+ + c2*(y+)^2+..."
            ),
            "continuity_coefficient": (
                "2*b2 + partial_xplus(a1) + partial_zplus(c1) = 0"
            ),
            "reynolds_stress": (
                "-<u'v'>+=-<a1*b2>*(y+)^3+O((y+)^4)"
            ),
            "dissipation": (
                "epsilon+=<a1^2+c1^2>+O(y+)"
            ),
            "ratio_coefficient": (
                "c_w=-<a1*b2>/<a1^2+c1^2> (using dU+/dy+|wall=1)"
            ),
        },
        "formula_series": {
            "pysr": wall_series(pysr["alpha"])["pysr_linear_tanh"],
            "m16": wall_series(m16["alpha"])["m16_squared_tanh"],
            "note": (
                "The M16 denominator generates a nonzero quartic term; writing "
                "only c_w*(y+)^3+O((y+)^5) is generically incorrect."
            ),
        },
        "aeh_refit_with_printed_practical_m16_triplet": {
            "fixed_alpha1_alpha2_alpha3": [a1, a2, a3],
            "per_Re_tau": rows,
            "regression": aeh,
        },
        "explicit_closure": (
            "F(y+,Re_tau)=tanh^2(0.111*y+)/[tanh(0.052*y+-0.443)+"
            "{alpha4_0+alpha5*ln(Re_tau/1000)}/y+]"
        ),
        "admissibility": domain,
        "high_Re_asymptotics": {
            "fixed_y_plus": "F ~ y+*tanh^2(alpha1*y+)/[alpha5*ln(Re_tau)] -> 0",
            "fixed_finite_Re_tau_then_y_plus_to_infinity": "F -> 1",
            "noncommuting_limits": {
                "lim_Re_then_lim_y": 0,
                "lim_y_then_lim_Re": 1,
            },
            "claim_boundary": (
                "Only Re_tau=180--5200 is fitted/validated; larger probes are "
                "unvalidated algebraic extrapolations."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "alpha4_0": aeh["alpha4_0"], "alpha5": aeh["alpha5"]}, indent=2))


if __name__ == "__main__":
    main()
