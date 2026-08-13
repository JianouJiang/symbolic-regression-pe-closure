#!/usr/bin/env python3
"""Exposure-aware, calibration-matched PySR--M16 paired comparison.

This reference implementation deliberately separates three phases.
``inventory`` reads paths and hashes only; it has no validation-array loader.
``fit`` opens only the five Lee--Moser training NPZ files and freezes two fits
made from the same multistart matrix. ``adjudicate`` consumes a later paired-row
file, checks completeness/roles/exposure, and applies a clustered paired-CI
rule.  Thus freezing blocks new coefficient adaptation without falsely turning
historically inspected data into an untouched lockbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
NODE = HERE
ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "fit_config.json"


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


def dataset_inventory() -> list[dict]:
    """Return metadata only. No scientific target array is opened here."""
    rows: list[dict] = []

    def add(dataset_id: str, source_family: str, flow_family: str, role: str,
            exposure: str, paths: Iterable[Path], availability: str = "ready",
            resolution: str | None = None) -> None:
        source_paths = list(paths)
        rows.append({
            "dataset_id": dataset_id,
            "source_family": source_family,
            "flow_family": flow_family,
            "role": role,
            "historical_exposure": exposure,
            "required": True,
            "availability": availability,
            "source_paths": [relative(path) for path in source_paths],
            "source_sha256": {
                relative(path): sha256(path) if path.is_file() else None
                for path in source_paths
            },
            "resolution": resolution,
        })

    for re_tau in [180, 550, 1000, 2000, 5200]:
        add(
            f"LM_channel_Re{re_tau}", "Lee_Moser_channel", "channel", "training",
            "training", [ROOT / "codes" / "results" / f"channel_Re{re_tau}.npz"],
        )
    spectra = ROOT / "codes" / "data_processing" / "spectra_data"
    for re_tau in [180, 550, 950, 2000]:
        add(
            f"UPM_channel_Re{re_tau}", "UPM_channel", "channel",
            "external_re_evaluation_in_domain", "used_in_original_model_selection",
            [spectra / f"UPM_Re{re_tau}_balance_kbal.dat"],
        )
    tbl = ROOT / "codes" / "results" / "kth_tbl_budgets"
    for re_theta in [670, 1000, 1410, 2000, 2540, 3030, 3270, 3630, 3970, 4060]:
        add(
            f"KTH_TBL_Retheta{re_theta}", "KTH_ZPG_TBL", "ZPG_TBL",
            "external_re_evaluation_in_domain", "used_in_original_model_selection",
            [tbl / f"bud_{re_theta:04d}_dns_k.prof"],
        )
    pipe = ROOT / "codes" / "results" / "el_khoury_pipe"
    for re_tau in [180, 360, 550, 1000]:
        add(
            f"ElKhoury_pipe_Re{re_tau}", "ElKhoury_pipe", "pipe",
            "external_re_evaluation_in_domain", "used_in_original_model_selection",
            [pipe / f"{re_tau}_{component}_Budget.dat" for component in ["RR", "TT", "ZZ"]],
        )
    add(
        "Yao_pipe_Re5200", "Yao_pipe", "pipe", "external_re_evaluation_in_domain",
        "reported_validation_not_known_used_for_selection",
        [ROOT / "related_papers" / "pipe_dns_budget_data" / "PIPE_Re5K_RSTE_k.dat"],
    )
    for case in ["b1n", "b2n", "m13n", "m16n", "m18n"]:
        add(
            f"Bobke_APG_{case}", "Bobke_APG", "equilibrium_APG_TBL",
            "external_re_evaluation_in_domain", "used_in_original_model_selection",
            [ROOT / "codes" / "results" / f"apg_tbl_kth_{case}.npz"],
            "blocked_missing_production_dissipation_fields",
            "Regenerate from the original APG.mat retaining Pk and Dk; do not infer targets.",
        )
    add(
        "Lardeau_APG_station_set", "Lardeau_recovering_APG", "recovering_APG_TBL",
        "external_re_evaluation_in_domain", "used_in_original_model_selection",
        [ROOT / "codes" / "data_processing" / "_download_cache" / "2d_budgets" /
         "curved_backstep_les" / "curvedbackstep_budgets_all.dat"],
        "blocked_station_list_and_wall_scaling_not_frozen",
        "Freeze exact submitted attached stations and independent wall quantities before expansion.",
    )
    add(
        "Balakumar_periodic_hill_station_set", "Balakumar_periodic_hill", "separated_APG",
        "external_re_evaluation_in_domain", "used_in_original_model_selection",
        [ROOT / "related_papers" / "apg_separated_budget_data"],
        "blocked_independent_wall_coordinate",
        "Outcome-dependent production-peak wall scaling is inadmissible; source u_tau independently.",
    )
    for re_tau in [93, 220, 500]:
        add(
            f"Couette_Re{re_tau}", "Lee_Moser_Couette", "Couette",
            "out_of_domain_diagnostic", "out_of_domain_diagnostic",
            [ROOT / "codes" / "results" / f"couette_Re{re_tau}.npz"],
        )
    return rows


def inventory_phase(config: dict) -> dict:
    rows = dataset_inventory()
    allowed = set(config["historical_exposure_labels"])
    for row in rows:
        if row["historical_exposure"] not in allowed:
            raise ValueError(f"unregistered exposure label: {row['dataset_id']}")
    return {
        "schema_version": "2.0",
        "phase": "metadata_and_hash_only_prefreeze",
        "validation_target_arrays_loaded": False,
        "rows": rows,
        "n_rows_or_unresolved_sets": len(rows),
        "n_release_blockers": sum(row["availability"] != "ready" for row in rows),
        "primary_role": config["primary_role"],
        "terminology": (
            "Historically inspected rows are external re-evaluation, not untouched held-out data."
        ),
    }


def formula(y_plus: np.ndarray, power: int, alpha: np.ndarray) -> np.ndarray:
    a1, a2, a3, a4 = np.asarray(alpha, dtype=float)
    y = np.asarray(y_plus, dtype=float)
    return np.tanh(a1 * y) ** power / (np.tanh(a2 * y - a3) + a4 / y)


def training_profiles(config: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    profiles = []
    mask_cfg = config["mask"]
    for re_tau in [180, 550, 1000, 2000, 5200]:
        path = ROOT / "codes" / "results" / f"channel_Re{re_tau}.npz"
        with np.load(path) as data:
            y = np.asarray(data["y_plus"], dtype=float)
            z = np.asarray(data["P_over_eps"], dtype=float)
            epsilon = np.asarray(data["dissipation"], dtype=float)
        y_max = min(mask_cfg["y_plus_max_absolute"], mask_cfg["y_plus_max_re_fraction"] * re_tau)
        mask = (
            (y > mask_cfg["y_plus_min_exclusive"]) & (y <= y_max)
            & (epsilon > mask_cfg["epsilon_min"])
            & (z > mask_cfg["p_over_epsilon_min"])
            & np.isfinite(y) & np.isfinite(z) & np.isfinite(epsilon)
        )
        profiles.append((f"LM_channel_Re{re_tau}", y[mask], z[mask]))
    return profiles


def common_start_matrix(config: dict) -> np.ndarray:
    cfg = config["matched_calibration"]
    bounds = np.asarray(cfg["bounds"], dtype=float)
    rng = np.random.default_rng(int(config["seed"]))
    unit = rng.random((int(cfg["multistarts"]), 4))
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def equal_profile_loss(alpha: np.ndarray, power: int,
                       profiles: list[tuple[str, np.ndarray, np.ndarray]], config: dict) -> float:
    dense_y = np.geomspace(1e-4, config["mask"]["y_plus_max_absolute"], 2048)
    denominator = np.tanh(alpha[1] * dense_y - alpha[2]) + alpha[3] / dense_y
    if not np.all(np.isfinite(denominator)) or float(np.min(denominator)) <= 1e-10:
        return 1e12
    losses = []
    for _dataset_id, y, target in profiles:
        prediction = formula(y, power, alpha)
        sse = float(np.sum((target - prediction) ** 2))
        sst = float(np.sum((target - np.mean(target)) ** 2))
        losses.append(sse / sst)
    return float(np.mean(losses))


def fit_family(power: int, starts: np.ndarray,
               profiles: list[tuple[str, np.ndarray, np.ndarray]], config: dict) -> dict:
    cfg = config["matched_calibration"]
    bounds = [tuple(pair) for pair in cfg["bounds"]]
    best: dict | None = None
    attempts = []
    for index, start in enumerate(starts):
        result = minimize(
            equal_profile_loss, start, args=(power, profiles, config), method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(cfg["max_iterations"]), "ftol": cfg["optimizer_ftol"]},
        )
        item = {
            "start_index": index,
            "start": start.tolist(),
            "alpha": result.x.tolist(),
            "loss": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message),
        }
        attempts.append(item)
        if best is None or item["loss"] < best["loss"]:
            best = item
    assert best is not None
    return {"power": power, "best": best, "attempts": attempts}


def fit_phase(config: dict) -> dict:
    profiles = training_profiles(config)
    starts = common_start_matrix(config)
    fits = {
        "linear_m1": fit_family(1, starts, profiles, config),
        "cubic_m2": fit_family(2, starts, profiles, config),
    }
    result = {
        "schema_version": "2.0",
        "phase": "training_only_common_start_fit",
        "validation_target_arrays_loaded": False,
        "training_dataset_ids": [item[0] for item in profiles],
        "start_matrix_sha256": canonical_hash(starts.tolist()),
        "same_start_matrix_for_both_arms": True,
        "loss": "equal-profile mean of SSE_profile/SST_profile",
        "fits": fits,
        "config_sha256": canonical_hash(config),
        "script_sha256": sha256(Path(__file__)),
    }
    result["freeze_sha256"] = canonical_hash(result)
    return result


def percentile_interval(values: np.ndarray, confidence: float) -> list[float]:
    alpha = 1.0 - confidence
    return [float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))]


def paired_cluster_bootstrap(rows: list[dict], config: dict) -> dict:
    """Equal-source-family outer bootstrap, profile bootstrap within family."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_family"]].append(row)
    families = sorted(groups)
    if not families:
        raise ValueError("primary recommendation population is empty")
    rng = np.random.default_rng(int(config["seed"]) + 991)
    n_boot = int(config["bootstrap_replicates"])
    draws_r2 = np.empty(n_boot)
    draws_rmse = np.empty(n_boot)
    for index in range(n_boot):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        family_r2 = []
        family_rmse = []
        for family in sampled_families:
            members = groups[str(family)]
            sampled_members = rng.choice(len(members), size=len(members), replace=True)
            family_r2.append(float(np.median([members[j]["delta_R2"] for j in sampled_members])))
            family_rmse.append(float(np.median([members[j]["relative_RMSE_change"] for j in sampled_members])))
        draws_r2[index] = float(np.median(family_r2))
        draws_rmse[index] = float(np.median(family_rmse))
    confidence = float(config["confidence_level"])
    point_r2 = float(np.median([np.median([r["delta_R2"] for r in members]) for members in groups.values()]))
    point_rmse = float(np.median([np.median([r["relative_RMSE_change"] for r in members]) for members in groups.values()]))
    return {
        "estimand": "median of source-family medians; equal source-family weight",
        "outer_resampling_unit": "source_family",
        "inner_resampling_unit": "profile",
        "n_source_families": len(families),
        "source_families": families,
        "delta_R2": {"point": point_r2, "two_sided_CI": percentile_interval(draws_r2, confidence)},
        "relative_RMSE_change": {"point": point_rmse, "two_sided_CI": percentile_interval(draws_rmse, confidence)},
    }


def family_summaries(rows: list[dict], config: dict) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_family"]].append(row)
    rng = np.random.default_rng(int(config["seed"]) + 1991)
    n_boot = int(config["bootstrap_replicates"])
    confidence = float(config["confidence_level"])
    output = []
    for family in sorted(groups):
        members = groups[family]
        r2_draw = np.empty(n_boot)
        rmse_draw = np.empty(n_boot)
        for index in range(n_boot):
            sample = rng.choice(len(members), size=len(members), replace=True)
            r2_draw[index] = float(np.median([members[j]["delta_R2"] for j in sample]))
            rmse_draw[index] = float(np.median([members[j]["relative_RMSE_change"] for j in sample]))
        output.append({
            "source_family": family,
            "n_profiles": len(members),
            "delta_R2_median": float(np.median([row["delta_R2"] for row in members])),
            "delta_R2_CI": percentile_interval(r2_draw, confidence),
            "relative_RMSE_change_median": float(np.median([row["relative_RMSE_change"] for row in members])),
            "relative_RMSE_change_CI": percentile_interval(rmse_draw, confidence),
        })
    return output


def adjudicate_phase(config: dict, paired_rows_path: Path, freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    expected_freeze_hash = freeze.get("freeze_sha256")
    without_hash = {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    if expected_freeze_hash != canonical_hash(without_hash):
        raise ValueError("freeze artifact hash does not verify")
    payload = json.loads(paired_rows_path.read_text())
    rows = payload["rows"]
    inventory = dataset_inventory()
    registry = {row["dataset_id"]: row for row in inventory}
    required_fields = {
        "dataset_id", "source_family", "flow_family", "role", "historical_exposure",
        "original_PySR", "M16", "matched_linear_m1", "matched_cubic_m2",
        "delta_R2", "relative_RMSE_change", "matched_delta_R2",
        "matched_relative_RMSE_change",
    }
    for row in rows:
        missing = required_fields.difference(row)
        if missing:
            raise ValueError(f"{row.get('dataset_id')}: missing {sorted(missing)}")
        meta = registry.get(row["dataset_id"])
        if meta is None:
            raise ValueError(f"unregistered dataset: {row['dataset_id']}")
        for field in ["source_family", "flow_family", "role", "historical_exposure"]:
            if row[field] != meta[field]:
                raise ValueError(f"{row['dataset_id']}: registry mismatch for {field}")
        for formula_name in ["original_PySR", "M16", "matched_linear_m1", "matched_cubic_m2"]:
            metric = row[formula_name]
            if not isinstance(metric, dict) or not {"R2", "RMSE", "n"}.issubset(metric):
                raise ValueError(f"{row['dataset_id']}: incomplete metrics for {formula_name}")
            if metric["n"] < 5 or not np.isfinite(metric["R2"]) or not np.isfinite(metric["RMSE"]):
                raise ValueError(f"{row['dataset_id']}: invalid metrics for {formula_name}")
        counts = {row[name]["n"] for name in ["original_PySR", "M16", "matched_linear_m1", "matched_cubic_m2"]}
        if len(counts) != 1:
            raise ValueError(f"{row['dataset_id']}: formula arms used different masks")
        exact_delta_r2 = row["M16"]["R2"] - row["original_PySR"]["R2"]
        exact_relative_rmse = row["M16"]["RMSE"] / row["original_PySR"]["RMSE"] - 1.0
        matched_delta_r2 = row["matched_cubic_m2"]["R2"] - row["matched_linear_m1"]["R2"]
        matched_relative_rmse = row["matched_cubic_m2"]["RMSE"] / row["matched_linear_m1"]["RMSE"] - 1.0
        checks = [
            ("delta_R2", exact_delta_r2),
            ("relative_RMSE_change", exact_relative_rmse),
            ("matched_delta_R2", matched_delta_r2),
            ("matched_relative_RMSE_change", matched_relative_rmse),
        ]
        for field, expected in checks:
            if not np.isclose(row[field], expected, rtol=1e-10, atol=1e-12):
                raise ValueError(f"{row['dataset_id']}: inconsistent {field}")
    primary_role = config["primary_role"]
    primary = [row for row in rows if row["role"] == primary_role]
    expected_primary = {
        row["dataset_id"] for row in inventory
        if row["role"] == primary_role and row["availability"] == "ready"
    }
    observed_primary = {row["dataset_id"] for row in primary}
    unavailable = [
        row for row in inventory
        if row["role"] == primary_role and row["availability"] != "ready"
    ]
    missing_ready = sorted(expected_primary.difference(observed_primary))
    unexpected = sorted(observed_primary.difference(expected_primary))
    release_blockers = [
        {"dataset_id": row["dataset_id"], "availability": row["availability"],
         "resolution": row["resolution"]} for row in unavailable
    ]
    release_blockers.extend({"dataset_id": item, "availability": "missing_paired_row"} for item in missing_ready)
    release_blockers.extend({"dataset_id": item, "availability": "unexpected_primary_row"} for item in unexpected)
    summary = paired_cluster_bootstrap(primary, config) if primary else None
    by_family = family_summaries(primary, config) if primary else []
    matched_primary = [
        {
            **row,
            "delta_R2": row["matched_delta_R2"],
            "relative_RMSE_change": row["matched_relative_RMSE_change"],
        }
        for row in primary
    ]
    matched_summary = paired_cluster_bootstrap(matched_primary, config) if matched_primary else None
    matched_by_family = family_summaries(matched_primary, config) if matched_primary else []
    thresholds = config["noninferiority"]
    family_gate_failures = []
    for item in by_family:
        if item["n_profiles"] >= int(thresholds["minimum_profiles_for_family_gate"]):
            if (item["delta_R2_CI"][0] < thresholds["delta_r2_margin"]
                    or item["relative_RMSE_change_CI"][1] > thresholds["relative_rmse_margin"]):
                family_gate_failures.append(item["source_family"])
    noninferior = bool(
        summary
        and summary["delta_R2"]["two_sided_CI"][0] >= thresholds["delta_r2_margin"]
        and summary["relative_RMSE_change"]["two_sided_CI"][1] <= thresholds["relative_rmse_margin"]
        and not family_gate_failures
    )
    permitted = not release_blockers
    recommendation = ("M16" if noninferior else "original_PySR") if permitted else None
    return {
        "schema_version": "2.0",
        "phase": "reported_form_adjudication",
        "freeze_sha256_verified": expected_freeze_hash,
        "primary_population": primary_role,
        "training_excluded_from_recommendation": True,
        "couette_excluded_from_recommendation": True,
        "historically_exposed_data_called_untouched": False,
        "n_primary_rows": len(primary),
        "summary": summary,
        "family_summaries": by_family,
        "matched_calibration_summary": matched_summary,
        "matched_calibration_family_summaries": matched_by_family,
        "family_gate_failures": family_gate_failures,
        "release_blockers": release_blockers,
        "recommendation_permitted": permitted,
        "reported_M16_noninferior": noninferior,
        "recommended_formula": recommendation,
        "rule": (
            "Recommend M16 only when the lower paired CI for delta R2 is >= -0.01, "
            "the upper paired CI for relative RMSE is <= 0.05, no eligible family gate "
            "fails, and the required suite is complete; otherwise recommend original PySR."
        ),
        "all_rows": rows,
    }


def write_result(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"phase": result["phase"], "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=["inventory", "fit", "adjudicate"], required=True)
    parser.add_argument("--paired-rows", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase == "inventory":
        result = inventory_phase(config)
        output = args.output or NODE / "results" / "prefreeze_inventory.json"
    elif args.phase == "fit":
        result = fit_phase(config)
        output = args.output or NODE / "results" / "matched_fit_freeze.json"
    else:
        if args.paired_rows is None or args.freeze is None:
            parser.error("adjudicate requires --paired-rows and --freeze")
        result = adjudicate_phase(config, args.paired_rows, args.freeze)
        output = args.output or NODE / "fit_decision.json"
    result.setdefault("script_sha256", sha256(Path(__file__)))
    result.setdefault("config_sha256", canonical_hash(config))
    write_result(result, output)


if __name__ == "__main__":
    main()
