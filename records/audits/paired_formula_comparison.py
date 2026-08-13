#!/usr/bin/env python3
"""Leakage-free reported-form comparison and matched-calibration scaffold.

The default ``manifest`` phase is read-only with respect to scientific data and
records which rows of the submitted validation suite can be regenerated from
local source arrays.  ``reported`` refuses to issue a recommendation while any
required row is unavailable unless ``--allow-incomplete`` is explicitly used.
The ``fit`` phase touches training channels only and freezes symmetric fits for
the m=1 and m=2 structural families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
NODE = HERE
ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "audit_config.json"


@dataclass
class Dataset:
    dataset_id: str
    family: str
    role: str
    source_label: str
    source_paths: list[Path]
    re_tau: float | None
    y_plus: np.ndarray
    target: np.ndarray
    epsilon: np.ndarray
    y_plus_max_override: float | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def numeric_rows(path: Path, min_columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("%", "#")):
            continue
        try:
            values = [float(token) for token in stripped.split()]
        except ValueError:
            continue
        if len(values) >= min_columns:
            rows.append(values)
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return np.asarray(rows, dtype=float)


def formula(y_plus: np.ndarray, power: int, alpha: np.ndarray) -> np.ndarray:
    y = np.asarray(y_plus, dtype=float)
    a1, a2, a3, a4 = np.asarray(alpha, dtype=float)
    return np.tanh(a1 * y) ** power / (
        np.tanh(a2 * y - a3) + a4 / y
    )


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    residual = target - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "n": int(target.size),
    }


def apply_mask(dataset: Dataset, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = config["mask"]
    if dataset.y_plus_max_override is not None:
        y_max = dataset.y_plus_max_override
    elif dataset.re_tau is not None:
        y_max = min(
            cfg["y_plus_max_absolute"],
            cfg["y_plus_max_re_fraction"] * dataset.re_tau,
        )
    else:
        raise ValueError(f"{dataset.dataset_id}: no Re_tau or explicit y+ maximum")
    mask = (
        (dataset.y_plus > cfg["y_plus_min_exclusive"])
        & (dataset.y_plus <= y_max)
        & (dataset.epsilon > cfg["epsilon_min"])
        & (dataset.target > cfg["p_over_epsilon_min"])
        & np.isfinite(dataset.y_plus)
        & np.isfinite(dataset.target)
        & np.isfinite(dataset.epsilon)
    )
    if mask.sum() < 5:
        raise ValueError(f"{dataset.dataset_id}: only {mask.sum()} points after mask")
    return dataset.y_plus[mask], dataset.target[mask], dataset.epsilon[mask]


def load_training_channels() -> list[Dataset]:
    output: list[Dataset] = []
    for re_tau in [180, 550, 1000, 2000, 5200]:
        path = ROOT / "codes" / "results" / f"channel_Re{re_tau}.npz"
        with np.load(path) as data:
            output.append(
                Dataset(
                    dataset_id=f"LM_channel_Re{re_tau}",
                    family="channel_training",
                    role="training",
                    source_label="Lee--Moser channel DNS",
                    source_paths=[path],
                    re_tau=float(re_tau),
                    y_plus=np.asarray(data["y_plus"], dtype=float),
                    target=np.asarray(data["P_over_eps"], dtype=float),
                    epsilon=np.asarray(data["dissipation"], dtype=float),
                )
            )
    return output


def load_upm() -> list[Dataset]:
    output: list[Dataset] = []
    base = ROOT / "codes" / "data_processing" / "spectra_data"
    for re_tau in [180, 550, 950, 2000]:
        path = base / f"UPM_Re{re_tau}_balance_kbal.dat"
        raw = numeric_rows(path, 4)
        epsilon = np.abs(raw[:, 2])
        output.append(
            Dataset(
                dataset_id=f"UPM_channel_Re{re_tau}",
                family="channel_independent",
                role="validation",
                source_label="Hoyas--Jimenez UPM channel DNS",
                source_paths=[path],
                re_tau=float(re_tau),
                y_plus=raw[:, 1],
                target=raw[:, 3] / np.maximum(epsilon, np.finfo(float).tiny),
                epsilon=epsilon,
            )
        )
    return output


def parse_re_tau_header(path: Path) -> float:
    pattern = re.compile(r"Re_\{\\tau\}\s*=\s*([0-9.]+)")
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    raise ValueError(f"Could not parse Re_tau in {path}")


def load_tbl() -> list[Dataset]:
    output: list[Dataset] = []
    base = ROOT / "codes" / "results" / "kth_tbl_budgets"
    for re_theta in [670, 1000, 1410, 2000, 2540, 3030, 3270, 3630, 3970, 4060]:
        path = base / f"bud_{re_theta:04d}_dns_k.prof"
        raw = numeric_rows(path, 9)
        re_tau = parse_re_tau_header(path)
        epsilon = np.abs(raw[:, 4])
        output.append(
            Dataset(
                dataset_id=f"KTH_TBL_Retheta{re_theta}",
                family="ZPG_TBL",
                role="validation",
                source_label="Schlatter--Orlu KTH ZPG TBL DNS",
                source_paths=[path],
                re_tau=re_tau,
                y_plus=raw[:, 1],
                target=raw[:, 3] / np.maximum(epsilon, np.finfo(float).tiny),
                epsilon=epsilon,
            )
        )
    return output


def load_el_khoury() -> list[Dataset]:
    output: list[Dataset] = []
    base = ROOT / "codes" / "results" / "el_khoury_pipe"
    for re_tau in [180, 360, 550, 1000]:
        paths = [base / f"{re_tau}_{component}_Budget.dat" for component in ["RR", "TT", "ZZ"]]
        components = [numeric_rows(path, 8) for path in paths]
        n = min(array.shape[0] for array in components)
        components = [array[:n] for array in components]
        y_plus = components[0][:, 1]
        production = 0.5 * sum(array[:, 2] for array in components)
        epsilon = -0.5 * sum(array[:, 7] for array in components)
        output.append(
            Dataset(
                dataset_id=f"ElKhoury_pipe_Re{re_tau}",
                family="pipe",
                role="validation",
                source_label="El Khoury et al. pipe DNS",
                source_paths=paths,
                re_tau=float(re_tau),
                y_plus=y_plus,
                target=production / np.maximum(epsilon, np.finfo(float).tiny),
                epsilon=epsilon,
            )
        )
    return output


def load_yao() -> list[Dataset]:
    path = ROOT / "related_papers" / "pipe_dns_budget_data" / "PIPE_Re5K_RSTE_k.dat"
    raw = numeric_rows(path, 9)
    epsilon = -raw[:, 8]
    return [
        Dataset(
            dataset_id="Yao_pipe_Re5200",
            family="pipe",
            role="validation",
            source_label="Yao et al. Texas Tech pipe DNS",
            source_paths=[path],
            re_tau=5200.0,
            y_plus=raw[:, 2],
            target=raw[:, 3] / np.maximum(epsilon, np.finfo(float).tiny),
            epsilon=epsilon,
        )
    ]


def load_couette() -> list[Dataset]:
    output: list[Dataset] = []
    for re_tau in [93, 220, 500]:
        path = ROOT / "codes" / "results" / f"couette_Re{re_tau}.npz"
        with np.load(path) as data:
            epsilon = np.asarray(data["dissipation"], dtype=float)
            production = -np.asarray(data["uv_plus"], dtype=float) * np.asarray(
                data["dUdy_plus"], dtype=float
            )
            output.append(
                Dataset(
                    dataset_id=f"Couette_Re{re_tau}",
                    family="Couette_OOD",
                    role="out_of_domain_diagnostic",
                    source_label="Lee--Moser plane Couette DNS",
                    source_paths=[path],
                    re_tau=float(re_tau),
                    y_plus=np.asarray(data["y_plus"], dtype=float),
                    target=production / np.maximum(epsilon, np.finfo(float).tiny),
                    epsilon=epsilon,
                )
            )
    return output


LOADERS: list[Callable[[], list[Dataset]]] = [
    load_training_channels,
    load_upm,
    load_tbl,
    load_el_khoury,
    load_yao,
    load_couette,
]


def locally_loadable_datasets() -> tuple[list[Dataset], list[dict]]:
    datasets: list[Dataset] = []
    errors: list[dict] = []
    for loader in LOADERS:
        try:
            datasets.extend(loader())
        except Exception as exc:  # manifest must preserve failures, not hide them
            errors.append({"loader": loader.__name__, "error": repr(exc)})
    return datasets, errors


def unresolved_required_rows() -> list[dict]:
    rows: list[dict] = []
    for case in ["b1n", "b2n", "m13n", "m16n", "m18n"]:
        source = ROOT / "codes" / "results" / f"apg_tbl_kth_{case}.npz"
        present_fields: list[str] = []
        if source.exists():
            with np.load(source, allow_pickle=True) as data:
                present_fields = list(data.files)
        rows.append(
            {
                "dataset_id": f"Bobke_APG_{case}",
                "family": "equilibrium_APG_TBL",
                "required": True,
                "availability": "blocked_missing_source_budget_fields",
                "source_path": relative(source),
                "source_sha256": sha256(source) if source.exists() else None,
                "present_fields": present_fields,
                "missing_fields": ["production", "dissipation"],
                "resolution": (
                    "Regenerate from the original APG.mat with Pk and Dk retained; "
                    "do not infer them from anisotropy-only arrays or old aggregate R2."
                ),
            }
        )
    rows.extend(
        [
            {
                "dataset_id": "Lardeau_extended_APG_stations",
                "family": "recovering_APG_TBL",
                "required": True,
                "availability": "source_present_station_manifest_not_yet_frozen",
                "source_path": (
                    "codes/data_processing/_download_cache/2d_budgets/"
                    "curved_backstep_les/curvedbackstep_budgets_all.dat"
                ),
                "resolution": (
                    "Freeze the exact submitted attached-station list and materialize "
                    "each y+, P/epsilon profile before either formula is evaluated."
                ),
            },
            {
                "dataset_id": "Balakumar_periodic_hill_stations",
                "family": "separated_APG",
                "required": True,
                "availability": "blocked_independent_wall_coordinate_provenance",
                "source_path": "related_papers/apg_separated_budget_data/",
                "resolution": (
                    "Provide an independently sourced u_tau/y+ mapping. The archived "
                    "production-peak calibration is outcome-dependent and cannot enter "
                    "the paired comparison as validation evidence."
                ),
            },
        ]
    )
    return rows


def manifest(config: dict) -> dict:
    datasets, loader_errors = locally_loadable_datasets()
    rows: list[dict] = []
    for dataset in datasets:
        try:
            y, target, _epsilon = apply_mask(dataset, config)
            status = "ready"
            error = None
        except Exception as exc:
            y = np.array([])
            target = np.array([])
            status = "blocked_mask_or_data"
            error = repr(exc)
        rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "family": dataset.family,
                "role": dataset.role,
                "required": True,
                "availability": status,
                "error": error,
                "n_after_mask": int(y.size),
                "actual_y_plus_range": (
                    [float(y.min()), float(y.max())] if y.size else None
                ),
                "target_range": (
                    [float(target.min()), float(target.max())] if target.size else None
                ),
                "source_paths": [relative(path) for path in dataset.source_paths],
                "source_sha256": {
                    relative(path): sha256(path) for path in dataset.source_paths
                },
            }
        )
    unresolved = unresolved_required_rows()
    return {
        "schema_version": "1.0",
        "mask": config["mask"],
        "loadable_rows": rows,
        "unresolved_required_rows": unresolved,
        "loader_errors": loader_errors,
        "n_ready": sum(row["availability"] == "ready" for row in rows),
        "n_unresolved_required": len(unresolved),
        "full_suite_ready": (
            not loader_errors
            and all(row["availability"] == "ready" for row in rows)
            and not unresolved
        ),
    }


def equal_profile_loss(alpha: np.ndarray, power: int, datasets: list[Dataset], config: dict) -> float:
    dense_y = np.geomspace(1e-4, config["mask"]["y_plus_max_absolute"], 2000)
    dense_denominator = np.tanh(alpha[1] * dense_y - alpha[2]) + alpha[3] / dense_y
    if np.min(dense_denominator) <= 1e-10:
        return 1e12
    losses: list[float] = []
    for dataset in datasets:
        y, target, _epsilon = apply_mask(dataset, config)
        prediction = formula(y, power, alpha)
        ss_res = np.sum((target - prediction) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        losses.append(float(ss_res / ss_tot))
    return float(np.mean(losses))


def fit_family(power: int, datasets: list[Dataset], config: dict) -> dict:
    cfg = config["matched_calibration"]
    bounds = np.asarray(cfg["bounds"], dtype=float)
    rng = np.random.default_rng(int(config["seed"]) + power)
    starts = rng.uniform(bounds[:, 0], bounds[:, 1], size=(int(cfg["multistarts"]), 4))
    reported_key = "pysr" if power == 1 else "m16"
    starts[0] = np.asarray(config["reported_formulas"][reported_key]["alpha"], dtype=float)
    best = None
    for index, start in enumerate(starts):
        result = minimize(
            equal_profile_loss,
            start,
            args=(power, datasets, config),
            method="L-BFGS-B",
            bounds=[tuple(pair) for pair in bounds],
            options={"maxiter": int(cfg["max_iterations"]), "ftol": 1e-14},
        )
        candidate = {
            "start_index": index,
            "alpha": result.x.tolist(),
            "loss_equal_profile_normalized_SSE": float(result.fun),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
        }
        if best is None or candidate["loss_equal_profile_normalized_SSE"] < best["loss_equal_profile_normalized_SSE"]:
            best = candidate
    assert best is not None
    return {"power": power, **best}


def fit_phase(config: dict) -> dict:
    training = load_training_channels()
    source_hashes = {
        relative(path): sha256(path)
        for dataset in training
        for path in dataset.source_paths
    }
    return {
        "schema_version": "1.0",
        "phase": "training_only_matched_calibration",
        "validation_was_loaded": False,
        "loss": (
            "mean over five Re_tau profiles of SSE_profile/SST_profile; "
            "identical masks, bounds, starts, and optimizer budget for m=1 and m=2"
        ),
        "source_sha256": source_hashes,
        "fits": {
            "linear_family_m1": fit_family(1, training, config),
            "cubic_family_m2": fit_family(2, training, config),
        },
    }


def reported_phase(config: dict, allow_incomplete: bool) -> dict:
    suite = manifest(config)
    if not suite["full_suite_ready"] and not allow_incomplete:
        raise RuntimeError(
            "Full suite is not traceable. Resolve every unresolved_required_row "
            "before evaluating or recommending a formula."
        )
    datasets, loader_errors = locally_loadable_datasets()
    if loader_errors:
        raise RuntimeError(f"Loader errors: {loader_errors}")
    formulas = config["reported_formulas"]
    rows: list[dict] = []
    for dataset in datasets:
        y, target, _epsilon = apply_mask(dataset, config)
        pred_pysr = formula(y, 1, np.asarray(formulas["pysr"]["alpha"], dtype=float))
        pred_m16 = formula(y, 2, np.asarray(formulas["m16"]["alpha"], dtype=float))
        pysr_metrics = metrics(target, pred_pysr)
        m16_metrics = metrics(target, pred_m16)
        rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "family": dataset.family,
                "role": dataset.role,
                "actual_y_plus_range": [float(y.min()), float(y.max())],
                "PySR": pysr_metrics,
                "M16": m16_metrics,
                "delta_R2_M16_minus_PySR": m16_metrics["R2"] - pysr_metrics["R2"],
                "delta_RMSE_M16_minus_PySR": m16_metrics["RMSE"] - pysr_metrics["RMSE"],
                "relative_RMSE_change_M16_over_PySR": (
                    m16_metrics["RMSE"] / pysr_metrics["RMSE"] - 1.0
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "phase": "reported_forms_primary_unconditional",
        "incomplete_suite_override": bool(allow_incomplete and not suite["full_suite_ready"]),
        "recommendation_permitted": bool(suite["full_suite_ready"]),
        "formula_constants": formulas,
        "rows": rows,
        "manifest": suite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=["manifest", "fit", "reported"], default="manifest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    if args.phase == "manifest":
        result = manifest(config)
        output = args.output or NODE / "results" / "dataset_manifest.json"
    elif args.phase == "fit":
        result = fit_phase(config)
        output = args.output or NODE / "results" / "matched_training_fit.json"
    else:
        result = reported_phase(config, args.allow_incomplete)
        output = args.output or NODE / "results" / "reported_formula_comparison.json"
    result["script_sha256"] = sha256(Path(__file__))
    result["config_sha256"] = sha256(args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"phase": args.phase, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
