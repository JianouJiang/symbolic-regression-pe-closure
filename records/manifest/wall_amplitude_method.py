#!/usr/bin/env python3
"""Training-only calibration of a wall-amplitude-orthogonalized M16 family.

The script never imports or opens a validation loader.  It estimates the
near-wall coefficient independently in each of the five Lee--Moser training
profiles, fixes the coefficient fold-by-fold, and compares two cubic models:

* hard M16: the submitted squared-tanh family with a4=a^2/c_w;
* M16-DA: a dual-activation numerator with a4=a*b/c_w.

M16-DA ties the exponential damping rate in the numerator to the denominator
transition rate.  It therefore retains four numerical constants
(a,b,d,c_w), has the exact coefficient c_w at the wall, and tends to unity in
the overlap limit.  Every trainable parameter is re-estimated in every
leave-one-Re_tau-out fold with the same deterministic start matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar


HERE = Path(__file__).resolve().parent
NODE = HERE.parent
ROOT = HERE.parents[3]
DEFAULT_CONFIG = HERE / "method_config.json"


@dataclass(frozen=True)
class Profile:
    re_tau: int
    dataset_id: str
    source_path: Path
    y_plus: np.ndarray
    target: np.ndarray
    epsilon: np.ndarray
    wall_coefficient: float
    wall_exponent: float
    wall_n: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def estimate_wall_coefficient(y: np.ndarray, z: np.ndarray, config: dict) -> tuple[float, float, int]:
    cfg = config["wall_coefficient_estimator"]
    mask = (
        (y > cfg["y_plus_min_exclusive"])
        & (y < cfg["y_plus_max_exclusive"])
        & (z > 0.0)
        & np.isfinite(y)
        & np.isfinite(z)
    )
    if int(mask.sum()) < 3:
        raise ValueError("fewer than three points in wall-coefficient window")
    exponent, log_coefficient = np.polyfit(np.log(y[mask]), np.log(z[mask]), 1)
    return float(np.exp(log_coefficient)), float(exponent), int(mask.sum())


def load_training_profiles(config: dict) -> list[Profile]:
    output: list[Profile] = []
    mask_cfg = config["mask"]
    for re_tau in config["training_re_tau"]:
        path = ROOT / "codes" / "results" / f"channel_Re{re_tau}.npz"
        with np.load(path) as data:
            y_all = np.asarray(data["y_plus"], dtype=float)
            z_all = np.asarray(data["P_over_eps"], dtype=float)
            eps_all = np.asarray(data["dissipation"], dtype=float)
        coefficient, exponent, wall_n = estimate_wall_coefficient(y_all, z_all, config)
        y_max = min(mask_cfg["y_plus_max_absolute"], mask_cfg["y_plus_max_re_fraction"] * re_tau)
        mask = (
            (y_all > mask_cfg["y_plus_min_exclusive"])
            & (y_all <= y_max)
            & (eps_all > mask_cfg["epsilon_min"])
            & (z_all > mask_cfg["p_over_epsilon_min"])
            & np.isfinite(y_all)
            & np.isfinite(z_all)
            & np.isfinite(eps_all)
        )
        if int(mask.sum()) < 5:
            raise ValueError(f"Re_tau={re_tau}: insufficient training points")
        output.append(
            Profile(
                re_tau=int(re_tau),
                dataset_id=f"LM_channel_Re{re_tau}",
                source_path=path,
                y_plus=y_all[mask],
                target=z_all[mask],
                epsilon=eps_all[mask],
                wall_coefficient=coefficient,
                wall_exponent=exponent,
                wall_n=wall_n,
            )
        )
    return output


def aggregate_cw(profiles: list[Profile], config: dict) -> float:
    method = config["wall_coefficient_estimator"]["profile_aggregation"]
    values = np.asarray([profile.wall_coefficient for profile in profiles], dtype=float)
    if method == "arithmetic mean":
        return float(np.mean(values))
    if method == "geometric mean":
        return float(np.exp(np.mean(np.log(values))))
    raise ValueError(f"unsupported c_w aggregation: {method}")


def hard_m16(y: np.ndarray, theta: np.ndarray, c_w: float) -> np.ndarray:
    a, b, d = np.asarray(theta, dtype=float)
    alpha4 = a * a / c_w
    return np.tanh(a * y) ** 2 / (np.tanh(b * y - d) + alpha4 / y)


def dual_activation(y: np.ndarray, theta: np.ndarray, c_w: float) -> np.ndarray:
    a, b, d = np.asarray(theta, dtype=float)
    alpha4 = a * b / c_w
    numerator = np.tanh(a * y) * (-np.expm1(-b * y))
    return numerator / (np.tanh(b * y - d) + alpha4 / y)


def formula(kind: str, y: np.ndarray, theta: np.ndarray, c_w: float) -> np.ndarray:
    if kind == "hard_anchored_M16":
        return hard_m16(y, theta, c_w)
    if kind == "dual_activation":
        return dual_activation(y, theta, c_w)
    raise ValueError(f"unknown model kind: {kind}")


def alpha4(kind: str, theta: np.ndarray, c_w: float) -> float:
    a, b, _d = np.asarray(theta, dtype=float)
    if kind == "hard_anchored_M16":
        return float(a * a / c_w)
    if kind == "dual_activation":
        return float(a * b / c_w)
    raise ValueError(kind)


def normalized_sse(target: np.ndarray, prediction: np.ndarray) -> float:
    sse = float(np.sum((target - prediction) ** 2))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    if not np.isfinite(sst) or sst <= 0.0:
        raise ValueError("non-positive profile SST")
    return sse / sst


def common_start_matrix(config: dict) -> np.ndarray:
    cfg = config["dual_activation"]
    bounds = np.asarray(cfg["bounds"], dtype=float)
    rng = np.random.default_rng(int(config["seed"]))
    unit = rng.random((int(cfg["multistarts"]), bounds.shape[0]))
    starts = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    return starts


def denominator_margin(kind: str, theta: np.ndarray, c_w: float, y_max: float) -> float:
    y = np.geomspace(1e-6, y_max, 4096)
    a, b, d = np.asarray(theta, dtype=float)
    a4 = alpha4(kind, theta, c_w)
    return float(np.min(np.tanh(b * y - d) + a4 / y))


def objective(theta: np.ndarray, kind: str, profiles: list[Profile], c_w: float, config: dict) -> float:
    margin = denominator_margin(kind, theta, c_w, config["mask"]["y_plus_max_absolute"])
    if not np.isfinite(margin) or margin <= 1e-10:
        return 1e12
    losses = [
        normalized_sse(profile.target, formula(kind, profile.y_plus, theta, c_w))
        for profile in profiles
    ]
    return float(np.mean(losses))


def fit_model(kind: str, profiles: list[Profile], c_w: float, starts: np.ndarray, config: dict) -> dict:
    cfg = config["dual_activation"] if kind == "dual_activation" else config["hard_anchored_M16"]
    bounds = [tuple(pair) for pair in cfg["bounds"]]
    attempts: list[dict] = []
    best: dict | None = None
    for start_index, start in enumerate(starts):
        result = minimize(
            objective,
            start,
            args=(kind, profiles, c_w, config),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(config["dual_activation"]["max_iterations"]),
                "ftol": float(config["dual_activation"]["optimizer_ftol"]),
            },
        )
        item = {
            "start_index": int(start_index),
            "start": [float(value) for value in start],
            "theta": [float(value) for value in result.x],
            "loss": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message),
        }
        attempts.append(item)
        if best is None or item["loss"] < best["loss"]:
            best = item
    assert best is not None
    theta = np.asarray(best["theta"], dtype=float)
    return {
        "kind": kind,
        "c_w_fixed": float(c_w),
        "best": best,
        "alpha4_dependent": alpha4(kind, theta, c_w),
        "denominator_min_dense": denominator_margin(
            kind, theta, c_w, config["mask"]["y_plus_max_absolute"]
        ),
        "attempts": attempts,
    }


def critical_alpha4(theta: np.ndarray) -> dict:
    """Exact denominator threshold max_y y*tanh(d-b*y), y in (0,d/b)."""
    _a, b, d = np.asarray(theta, dtype=float)
    upper = float(d / b)
    result = minimize_scalar(
        lambda y: -float(y * np.tanh(d - b * y)),
        bounds=(np.finfo(float).eps, upper),
        method="bounded",
    )
    return {
        "alpha4_critical": float(-result.fun),
        "y_plus_at_critical": float(result.x),
        "optimization_success": bool(result.success),
    }


def asymptotic_contracts(kind: str, theta: np.ndarray, c_w: float) -> dict:
    a, b, d = np.asarray(theta, dtype=float)
    a4 = alpha4(kind, theta, c_w)
    y_small = np.geomspace(1e-7, 1e-4, 32)
    coefficient_numeric = float(np.median(formula(kind, y_small, theta, c_w) / y_small**3))
    if kind == "dual_activation":
        quartic = float(-0.5 * b * c_w + np.tanh(d) * c_w**2 / (a * b))
        leading_identity = "a*b/alpha4=c_w"
    else:
        quartic = float(np.tanh(d) * c_w**2 / (a * a))
        leading_identity = "a^2/alpha4=c_w"
    critical = critical_alpha4(theta)
    return {
        "wall_exponent": 3,
        "leading_identity": leading_identity,
        "c_w_target": float(c_w),
        "c_w_numeric_from_F_over_y3": coefficient_numeric,
        "c_w_relative_numeric_error": coefficient_numeric / c_w - 1.0,
        "quartic_coefficient": quartic,
        "overlap_expansion": f"F=1-{a4:.12g}/y+O(1/y^2)",
        "F_at_y_plus_1e6": float(formula(kind, np.asarray([1e6]), theta, c_w)[0]),
        "alpha4": a4,
        **critical,
        "alpha4_admissibility_margin": float(a4 - critical["alpha4_critical"]),
    }


def leave_one_re_out(profiles: list[Profile], starts: np.ndarray, config: dict) -> dict:
    rows: list[dict] = []
    for held_index, held in enumerate(profiles):
        training = [profile for index, profile in enumerate(profiles) if index != held_index]
        c_w_fold = aggregate_cw(training, config)
        row = {
            "held_dataset_id": held.dataset_id,
            "held_re_tau": held.re_tau,
            "c_w_from_other_four": c_w_fold,
            "held_c_w": held.wall_coefficient,
            "held_c_w_relative_difference": c_w_fold / held.wall_coefficient - 1.0,
            "models": {},
        }
        for kind in ["hard_anchored_M16", "dual_activation"]:
            fit = fit_model(kind, training, c_w_fold, starts, config)
            theta = np.asarray(fit["best"]["theta"], dtype=float)
            held_prediction = formula(kind, held.y_plus, theta, c_w_fold)
            row["models"][kind] = {
                "theta": fit["best"]["theta"],
                "alpha4_dependent": fit["alpha4_dependent"],
                "training_loss": fit["best"]["loss"],
                "held_normalized_sse": normalized_sse(held.target, held_prediction),
                "optimizer_winning_start_index": fit["best"]["start_index"],
                "denominator_min_dense": fit["denominator_min_dense"],
            }
        rows.append(row)
    hard = np.asarray([row["models"]["hard_anchored_M16"]["held_normalized_sse"] for row in rows])
    dual = np.asarray([row["models"]["dual_activation"]["held_normalized_sse"] for row in rows])
    return {
        "rows": rows,
        "summary": {
            "hard_anchored_M16_mean_held_loss": float(np.mean(hard)),
            "dual_activation_mean_held_loss": float(np.mean(dual)),
            "dual_relative_mean_loss_change": float(np.mean(dual) / np.mean(hard) - 1.0),
            "dual_wins": int(np.sum(dual < hard)),
            "n_folds": int(len(rows)),
        },
    }


def run(config: dict) -> dict:
    profiles = load_training_profiles(config)
    starts = common_start_matrix(config)
    c_w_all = aggregate_cw(profiles, config)
    final_fits = {
        kind: fit_model(kind, profiles, c_w_all, starts, config)
        for kind in ["hard_anchored_M16", "dual_activation"]
    }
    for kind, fit in final_fits.items():
        theta = np.asarray(fit["best"]["theta"], dtype=float)
        fit["asymptotic_contracts"] = asymptotic_contracts(kind, theta, c_w_all)
    loo = leave_one_re_out(profiles, starts, config)
    result = {
        "schema_version": "3.0",
        "phase": "training_only_wall_amplitude_orthogonalization",
        "validation_target_arrays_loaded": False,
        "idea": (
            "Fix c_w independently and share the dissipation-activation rate between "
            "the numerator and denominator so finite-y shape can be calibrated without "
            "moving the cubic wall amplitude."
        ),
        "epistemic_status": (
            "closure-motivated candidate followed by DNS calibration; not a strict "
            "first-principles derivation and not parameter free"
        ),
        "training_source_sha256": {
            relative(profile.source_path): sha256(profile.source_path) for profile in profiles
        },
        "training_profiles": [
            {
                "dataset_id": profile.dataset_id,
                "re_tau": profile.re_tau,
                "n_fit": int(profile.y_plus.size),
                "actual_y_plus_range": [float(np.min(profile.y_plus)), float(np.max(profile.y_plus))],
                "wall_coefficient": profile.wall_coefficient,
                "wall_exponent": profile.wall_exponent,
                "wall_n": profile.wall_n,
            }
            for profile in profiles
        ],
        "c_w_training_aggregate": c_w_all,
        "start_matrix": starts.tolist(),
        "start_matrix_sha256": canonical_hash(starts.tolist()),
        "same_start_matrix_all_models_and_folds": True,
        "all_trainable_parameters_refit_in_every_fold": True,
        "loss": "equal-profile mean of SSE_profile/SST_profile",
        "final_fits": final_fits,
        "leave_one_re_tau_out": loo,
        "config_sha256": canonical_hash(config),
        "script_sha256": sha256(Path(__file__)),
    }
    result["freeze_sha256"] = canonical_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=NODE / "results" / "wall_amplitude_fit_freeze.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "freeze_sha256": result["freeze_sha256"],
                "c_w": result["c_w_training_aggregate"],
                "dual_theta": result["final_fits"]["dual_activation"]["best"]["theta"],
                "loo": result["leave_one_re_tau_out"]["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
