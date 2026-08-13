#!/usr/bin/env python3
"""Evaluate printed PySR, printed M16, and frozen cubic candidates.

The exact printed-form comparison is unconditional.  Candidate comparison is
secondary and cannot hide an adverse printed-M16 row.  All 179 frozen rows are
materialized with one common mask per row.  Only 19 intended-domain external
rows vote in the practical recommendation; APG boundary cases, recovering and
separated stations, training channels, and Couette remain visible as separate
sensitivity populations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
NODE = HERE
ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "method_config.json"
DEFAULT_FIT = HERE / "wall_amplitude_fit_freeze.json"
DEFAULT_MANIFEST = HERE / "profile_manifest_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LOADERS = load_module(
    "retained_loaders",
    ROOT / "records" / "audits" / "paired_formula_comparison.py",
)
METHOD = load_module("wall_method", HERE / "wall_amplitude_method.py")


def verify_freeze(payload: dict, hash_field: str) -> str:
    expected = payload.get(hash_field)
    if not expected:
        raise ValueError(f"missing {hash_field}")
    body = {key: value for key, value in payload.items() if key != hash_field}
    observed = canonical_hash(body)
    if observed != expected:
        raise ValueError(f"{hash_field} does not verify: {observed} != {expected}")
    return expected


def verify_manifest_sources(manifest: dict) -> dict[str, str]:
    expected: dict[str, str] = {}
    for row in manifest["rows"]:
        for path, digest in row["source"]["sha256"].items():
            if path in expected and expected[path] != digest:
                raise ValueError(f"inconsistent frozen hash for {path}")
            expected[path] = digest
    observed: dict[str, str] = {}
    for rel_path, digest in sorted(expected.items()):
        path = ROOT / rel_path
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"source changed after freeze: {rel_path}")
        observed[rel_path] = actual
    return observed


def printed_formula(y: np.ndarray, power: int, alpha: np.ndarray) -> np.ndarray:
    a1, a2, a3, a4 = np.asarray(alpha, dtype=float)
    return np.tanh(a1 * y) ** power / (np.tanh(a2 * y - a3) + a4 / y)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    residual = target - prediction
    sse = float(np.sum(residual**2))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    if target.size < 5 or not np.isfinite(sse) or not np.isfinite(sst) or sst <= 0:
        raise ValueError("invalid metric input")
    return {
        "R2": 1.0 - sse / sst,
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "n": int(target.size),
    }


def apply_common_mask(
    y: np.ndarray,
    target: np.ndarray,
    epsilon: np.ndarray,
    config: dict,
    re_tau: float | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    target = np.asarray(target, dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    if not (y.shape == target.shape == epsilon.shape):
        raise ValueError("y, target, and epsilon shapes differ")
    cfg = config["mask"]
    if re_tau is None:
        y_max = np.full_like(y, cfg["y_plus_max_absolute"], dtype=float)
    else:
        re_array = np.asarray(re_tau, dtype=float)
        if re_array.ndim == 0:
            re_array = np.full_like(y, float(re_array), dtype=float)
        if re_array.shape != y.shape:
            raise ValueError("Re_tau shape differs from profile")
        y_max = np.minimum(cfg["y_plus_max_absolute"], cfg["y_plus_max_re_fraction"] * re_array)
    mask = (
        (y > cfg["y_plus_min_exclusive"])
        & (y <= y_max)
        & (epsilon > cfg["epsilon_min"])
        & (target > cfg["p_over_epsilon_min"])
        & np.isfinite(y)
        & np.isfinite(target)
        & np.isfinite(epsilon)
    )
    if int(mask.sum()) < 5:
        raise ValueError(f"only {int(mask.sum())} points after common mask")
    return y[mask], target[mask], epsilon[mask]


def candidate_definitions(config: dict, fit: dict) -> dict:
    dual_fit = fit["final_fits"]["dual_activation"]
    hard_fit = fit["final_fits"]["hard_anchored_M16"]
    return {
        "original_PySR": {
            "kind": "printed",
            "power": 1,
            "alpha": config["reported_formulas"]["original_PySR"]["alpha"],
        },
        "printed_M16": {
            "kind": "printed",
            "power": 2,
            "alpha": config["reported_formulas"]["printed_M16"]["alpha"],
        },
        "hard_anchored_M16": {
            "kind": "hard_anchored_M16",
            "theta": hard_fit["best"]["theta"],
            "c_w": hard_fit["c_w_fixed"],
            "alpha4_dependent": hard_fit["alpha4_dependent"],
        },
        "M16_DA_wall_anchored": {
            "kind": "dual_activation",
            "theta": dual_fit["best"]["theta"],
            "c_w": dual_fit["c_w_fixed"],
            "alpha4_dependent": dual_fit["alpha4_dependent"],
        },
    }


def prediction(y: np.ndarray, definition: dict) -> np.ndarray:
    if definition["kind"] == "printed":
        return printed_formula(y, int(definition["power"]), np.asarray(definition["alpha"], dtype=float))
    return METHOD.formula(
        definition["kind"],
        y,
        np.asarray(definition["theta"], dtype=float),
        float(definition["c_w"]),
    )


def evaluate_arrays(
    meta: dict,
    y: np.ndarray,
    target: np.ndarray,
    epsilon: np.ndarray,
    re_tau: float | np.ndarray | None,
    config: dict,
    candidates: dict,
    extras: dict | None = None,
) -> dict:
    y_masked, target_masked, _epsilon_masked = apply_common_mask(
        y, target, epsilon, config, re_tau
    )
    values = {
        name: metrics(target_masked, prediction(y_masked, definition))
        for name, definition in candidates.items()
    }
    counts = {metric["n"] for metric in values.values()}
    if len(counts) != 1:
        raise AssertionError("candidate arms used different masks")
    baseline = values["original_PySR"]
    comparisons = {}
    for name, metric in values.items():
        if name == "original_PySR":
            continue
        comparisons[name] = {
            "delta_R2": metric["R2"] - baseline["R2"],
            "delta_RMSE": metric["RMSE"] - baseline["RMSE"],
            "relative_RMSE_change": metric["RMSE"] / baseline["RMSE"] - 1.0,
        }
    row = {
        "dataset_id": meta["dataset_id"],
        "source_family": meta["source_family"],
        "flow_family": meta["flow_family"],
        "role": meta["role"],
        "historical_exposure": meta["historical_exposure"],
        "actual_y_plus_range": [float(np.min(y_masked)), float(np.max(y_masked))],
        "metrics": values,
        "versus_original_PySR": comparisons,
    }
    if extras:
        row["profile_metadata"] = extras
    return row


def regular_rows(registry: dict[str, dict], config: dict, candidates: dict) -> list[dict]:
    datasets, loader_errors = LOADERS.locally_loadable_datasets()
    if loader_errors:
        raise RuntimeError(loader_errors)
    rows = []
    for dataset in datasets:
        meta = registry.get(dataset.dataset_id)
        if meta is None:
            raise KeyError(f"regular loader row absent from manifest: {dataset.dataset_id}")
        # Retained loader and the new evaluator use the same mask contract.
        y, target, epsilon = LOADERS.apply_mask(dataset, config)
        rows.append(
            evaluate_arrays(
                meta,
                y,
                target,
                epsilon,
                re_tau=None,
                config={**config, "mask": {**config["mask"], "y_plus_max_absolute": float(np.max(y))}},
                candidates=candidates,
                extras={"retained_loader_role": dataset.role},
            )
        )
    return rows


def bobke_rows(registry: dict[str, dict], config: dict, candidates: dict) -> list[dict]:
    mat_path = NODE / "source_cache" / "APG.mat"
    rows: list[dict] = []
    for case in ["b1n", "b2n", "m13n", "m16n", "m18n"]:
        dataset_id = f"Bobke_APG_{case}"
        meta = registry[dataset_id]
        case_data = loadmat(mat_path, variable_names=[case])[case]
        y_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        epsilon_parts: list[np.ndarray] = []
        re_parts: list[np.ndarray] = []
        beta_values: list[float] = []
        for station_index in range(case_data.shape[1]):
            station = case_data[0, station_index]
            y = np.asarray(station["y"], dtype=float).ravel()
            production = np.asarray(station["Pk"], dtype=float).ravel()
            dissipation_signed = np.asarray(station["Dk"], dtype=float).ravel()
            u_tau = float(np.asarray(station["ut"]).ravel()[0])
            nu = float(np.asarray(station["nu"]).ravel()[0])
            re_tau = float(np.asarray(station["Ret"]).ravel()[0])
            beta_values.append(float(np.asarray(station["beta"]).ravel()[0]))
            if u_tau <= 0.0 or nu <= 0.0:
                continue
            scale = u_tau**4 / nu
            epsilon = -dissipation_signed / scale
            production_plus = production / scale
            target = np.divide(
                production_plus,
                epsilon,
                out=np.full_like(production_plus, np.nan),
                where=epsilon > 0.0,
            )
            y_plus = y * u_tau / nu
            y_parts.append(y_plus)
            target_parts.append(target)
            epsilon_parts.append(epsilon)
            re_parts.append(np.full_like(y_plus, re_tau))
        rows.append(
            evaluate_arrays(
                meta,
                np.concatenate(y_parts),
                np.concatenate(target_parts),
                np.concatenate(epsilon_parts),
                np.concatenate(re_parts),
                config,
                candidates,
                extras={
                    "case": case,
                    "n_source_stations": int(case_data.shape[1]),
                    "beta_range_actual": [float(np.min(beta_values)), float(np.max(beta_values))],
                    "budget_fields": ["Pk", "Dk"],
                },
            )
        )
        del case_data
    return rows


def lardeau_rows(registry: dict[str, dict], config: dict, candidates: dict) -> list[dict]:
    path = ROOT / "codes" / "results" / "bfs_2d_budget.npz"
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=float)
        y_plus = np.asarray(data["y_plus"], dtype=float)
        target = np.asarray(data["P_over_eps"], dtype=float)
        epsilon = np.asarray(data["dissipation"], dtype=float)
    unique_x = np.unique(x)
    rows: list[dict] = []
    metas = [meta for meta in registry.values() if meta["source_family"] == "Lardeau_recovering_APG"]
    for meta in sorted(metas, key=lambda item: item["selector"]["x_over_H"]):
        requested = float(meta["selector"]["x_over_H"])
        selected = float(unique_x[int(np.argmin(np.abs(unique_x - requested)))])
        mask = np.isclose(x, selected, rtol=0.0, atol=1e-9)
        rows.append(
            evaluate_arrays(
                meta,
                y_plus[mask],
                target[mask],
                epsilon[mask],
                float(meta["selector"]["re_tau_local"]),
                config,
                candidates,
                extras={
                    "requested_x_over_H": requested,
                    "selected_x_over_H": selected,
                    "absolute_x_mismatch": abs(selected - requested),
                    "beta_clauser": float(meta["selector"]["beta_clauser"]),
                    "dissipation_caveat": "LES budget residual",
                },
            )
        )
    return rows


def periodic_hill_rows(registry: dict[str, dict], config: dict, candidates: dict) -> list[dict]:
    path = ROOT / "codes" / "results" / "pehill_2d_budget.npz"
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=float).reshape(128, 196)
        y_plus = np.asarray(data["y_plus"], dtype=float).reshape(128, 196)
        target = np.asarray(data["P_over_eps"], dtype=float).reshape(128, 196)
        epsilon = np.asarray(data["dissipation"], dtype=float).reshape(128, 196)
    bottom_x = x[0]
    rows: list[dict] = []
    metas = [meta for meta in registry.values() if meta["source_family"] == "TMBWG_periodic_hill"]
    for meta in sorted(metas, key=lambda item: item["selector"]["x_over_H"]):
        requested = float(meta["selector"]["x_over_H"])
        column = int(np.argmin(np.abs(bottom_x - requested)))
        rows.append(
            evaluate_arrays(
                meta,
                y_plus[:, column],
                target[:, column],
                epsilon[:, column],
                None,
                config,
                candidates,
                extras={
                    "requested_x_over_H": requested,
                    "column_index": column,
                    "bottom_cell_x_over_H": float(bottom_x[column]),
                    "absolute_x_mismatch": abs(float(bottom_x[column]) - requested),
                    "wall_coordinate": "independent digitized Cf, not production-peak calibration",
                },
            )
        )
    return rows


def source_family_summaries(rows: list[dict], candidate: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_family"]].append(row)
    output = []
    for family in sorted(groups):
        members = groups[family]
        comparisons = [row["versus_original_PySR"][candidate] for row in members]
        output.append(
            {
                "source_family": family,
                "n_rows": len(members),
                "median_delta_R2": float(np.median([item["delta_R2"] for item in comparisons])),
                "median_relative_RMSE_change": float(
                    np.median([item["relative_RMSE_change"] for item in comparisons])
                ),
                "R2_wins": int(sum(item["delta_R2"] > 0.0 for item in comparisons)),
                "RMSE_wins": int(sum(item["relative_RMSE_change"] < 0.0 for item in comparisons)),
            }
        )
    return output


def paired_family_bootstrap(rows: list[dict], candidate: str, config: dict) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_family"]].append(row)
    families = sorted(groups)
    rng = np.random.default_rng(int(config["seed"]) + sum(ord(char) for char in candidate))
    n_boot = int(config["uncertainty"]["bootstrap_replicates"])
    r2_draw = np.empty(n_boot)
    rmse_draw = np.empty(n_boot)
    for index in range(n_boot):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        family_r2 = []
        family_rmse = []
        for family in sampled_families:
            members = groups[str(family)]
            sampled = rng.integers(0, len(members), size=len(members))
            comparisons = [members[item]["versus_original_PySR"][candidate] for item in sampled]
            family_r2.append(float(np.median([value["delta_R2"] for value in comparisons])))
            family_rmse.append(
                float(np.median([value["relative_RMSE_change"] for value in comparisons]))
            )
        r2_draw[index] = float(np.median(family_r2))
        rmse_draw[index] = float(np.median(family_rmse))
    confidence = float(config["uncertainty"]["confidence_level"])
    alpha = 1.0 - confidence
    summaries = source_family_summaries(rows, candidate)
    return {
        "status": "descriptive because the number of independent source families is small",
        "n_source_families": len(families),
        "source_families": families,
        "estimand": "median of source-family medians",
        "delta_R2": {
            "point": float(np.median([item["median_delta_R2"] for item in summaries])),
            "two_sided_CI": [
                float(np.quantile(r2_draw, alpha / 2.0)),
                float(np.quantile(r2_draw, 1.0 - alpha / 2.0)),
            ],
        },
        "relative_RMSE_change": {
            "point": float(np.median([item["median_relative_RMSE_change"] for item in summaries])),
            "two_sided_CI": [
                float(np.quantile(rmse_draw, alpha / 2.0)),
                float(np.quantile(rmse_draw, 1.0 - alpha / 2.0)),
            ],
        },
    }


def leave_one_source_family_out(rows: list[dict], candidate: str) -> list[dict]:
    families = sorted({row["source_family"] for row in rows})
    output = []
    for omitted in families:
        retained = [row for row in rows if row["source_family"] != omitted]
        summaries = source_family_summaries(retained, candidate)
        output.append(
            {
                "omitted_source_family": omitted,
                "n_retained_families": len(summaries),
                "median_of_family_median_delta_R2": float(
                    np.median([item["median_delta_R2"] for item in summaries])
                ),
                "median_of_family_median_relative_RMSE_change": float(
                    np.median([item["median_relative_RMSE_change"] for item in summaries])
                ),
            }
        )
    return output


def candidate_decision(rows: list[dict], candidate: str, definition: dict, config: dict) -> dict:
    summaries = source_family_summaries(rows, candidate)
    limits = config["decision"]
    point_r2 = float(np.median([item["median_delta_R2"] for item in summaries]))
    point_rmse = float(np.median([item["median_relative_RMSE_change"] for item in summaries]))
    family_failures = [
        item["source_family"]
        for item in summaries
        if item["median_delta_R2"] < limits["delta_r2_margin"]
        or item["median_relative_RMSE_change"] > limits["relative_rmse_margin"]
    ]
    if candidate in {"hard_anchored_M16", "M16_DA_wall_anchored"}:
        c_w_error = 0.0
        cubic = True
    elif candidate == "printed_M16":
        alpha = np.asarray(definition["alpha"], dtype=float)
        target = float(config.get("_c_w_training_aggregate", math.nan))
        c_w_error = abs((alpha[0] ** 2 / alpha[3]) / target - 1.0)
        cubic = True
    else:
        c_w_error = math.inf
        cubic = False
    eligible = bool(
        point_r2 >= limits["delta_r2_margin"]
        and point_rmse <= limits["relative_rmse_margin"]
        and (not limits["require_all_primary_source_family_medians_within_margins"] or not family_failures)
        and (not limits["require_cubic_wall_exponent"] or cubic)
        and c_w_error <= limits["wall_coefficient_relative_error_max"]
    )
    return {
        "candidate": candidate,
        "eligible": eligible,
        "median_of_family_median_delta_R2": point_r2,
        "median_of_family_median_relative_RMSE_change": point_rmse,
        "family_gate_failures": family_failures,
        "cubic_wall_exponent": cubic,
        "wall_coefficient_relative_error": c_w_error,
        "family_summaries": summaries,
        "descriptive_cluster_bootstrap": paired_family_bootstrap(rows, candidate, config),
        "leave_one_source_family_out": leave_one_source_family_out(rows, candidate),
    }


def run(config: dict, fit: dict, manifest: dict) -> dict:
    fit_hash = verify_freeze(fit, "freeze_sha256")
    manifest_hash = verify_freeze(manifest, "manifest_sha256")
    config_hash = canonical_hash(config)
    if fit["config_sha256"] != config_hash or manifest["config_sha256"] != config_hash:
        raise ValueError("current config does not match both frozen artifacts")
    verified_sources = verify_manifest_sources(manifest)
    registry = {row["dataset_id"]: row for row in manifest["rows"]}
    candidates = candidate_definitions(config, fit)
    config = {**config, "_c_w_training_aggregate": fit["c_w_training_aggregate"]}
    rows = []
    rows.extend(regular_rows(registry, config, candidates))
    rows.extend(bobke_rows(registry, config, candidates))
    rows.extend(lardeau_rows(registry, config, candidates))
    rows.extend(periodic_hill_rows(registry, config, candidates))
    observed_ids = {row["dataset_id"] for row in rows}
    expected_ids = set(registry)
    if observed_ids != expected_ids:
        raise ValueError(
            f"materialized IDs differ from manifest; missing={sorted(expected_ids-observed_ids)}, "
            f"unexpected={sorted(observed_ids-expected_ids)}"
        )
    primary_role = config["uncertainty"]["primary_role"]
    primary = [row for row in rows if row["role"] == primary_role]
    decisions = {
        name: candidate_decision(primary, name, definition, config)
        for name, definition in candidates.items()
        if name != "original_PySR"
    }
    eligible = [name for name in ["M16_DA_wall_anchored", "hard_anchored_M16", "printed_M16"] if decisions[name]["eligible"]]
    recommendation = eligible[0] if eligible else "original_PySR"
    role_summaries = {}
    for role in sorted({row["role"] for row in rows}):
        subset = [row for row in rows if row["role"] == role]
        role_summaries[role] = {
            "n_rows": len(subset),
            "source_families": sorted({row["source_family"] for row in subset}),
            "comparisons": {
                name: source_family_summaries(subset, name)
                for name in candidates
                if name != "original_PySR"
            },
        }
    return {
        "schema_version": "3.0",
        "phase": "complete_profile_level_paired_comparison",
        "fit_freeze_sha256_verified": fit_hash,
        "manifest_sha256_verified": manifest_hash,
        "config_sha256_verified": config_hash,
        "n_unique_source_files_verified": len(verified_sources),
        "n_rows": len(rows),
        "n_primary_rows": len(primary),
        "candidate_definitions": candidates,
        "exact_printed_comparison_unconditional": True,
        "primary_population": primary_role,
        "stress_tests_excluded_from_recommendation_but_reported": True,
        "decisions": decisions,
        "recommended_formula": recommendation,
        "recommendation_rule": (
            "Choose the first eligible cubic candidate in the prospective order "
            "M16-DA, hard-anchored M16, printed M16; otherwise original PySR. "
            "Eligibility requires primary equal-source-family R2/RMSE margins, every "
            "primary source-family median within those margins, cubic wall order, and "
            "wall-coefficient error <=15%. Bootstrap intervals are descriptive; direct "
            "family gates and leave-one-family-out results carry the small-family audit."
        ),
        "role_summaries": role_summaries,
        "all_rows": rows,
        "script_sha256": sha256(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=HERE / "full_suite_evaluation.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    fit = json.loads(args.fit.read_text())
    manifest = json.loads(args.manifest.read_text())
    result = run(config, fit, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    compact = {
        "output": str(args.output),
        "n_rows": result["n_rows"],
        "n_primary_rows": result["n_primary_rows"],
        "recommended_formula": result["recommended_formula"],
        "decisions": {
            name: {
                "eligible": item["eligible"],
                "delta_R2": item["median_of_family_median_delta_R2"],
                "relative_RMSE_change": item["median_of_family_median_relative_RMSE_change"],
                "family_gate_failures": item["family_gate_failures"],
            }
            for name, item in result["decisions"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
